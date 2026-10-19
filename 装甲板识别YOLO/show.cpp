#include "show.h"

// ==========================================
// 模块 1 的实现：在这里给全局变量赋值
// ==========================================
// 对应 图3.md 真值数据 (相机光学系下)
vector<GroundTruth> gt_list = {
    {1, -0.179673, +0.053321, +1.746903, +115.614, -61.812, -9.306, "armor_1(背面)"},
    {2, +0.102393, +0.059418, +1.835529, +124.132, -55.870, -36.558, "armor_2(背面)"},
    {3, +0.189233, +0.070265, +1.553395, +85.635, -61.770, -9.327, "armor_3(正面)"},
    {4, -0.092414, +0.092079, +1.467008, +73.077, -60.945, +21.927, "armor_4(正面)"}
};
vector<Point3f> obj_points_yolo = {
    {-117.5f, -27.5f, 0}, // 左上 (0)
    {-117.5f,  27.5f, 0}, // 左下 (1)
    { 117.5f,  27.5f, 0}, // 右下 (2)
    { 117.5f, -27.5f, 0}  // 右上 (3)
};

Mat camera_matrix = (Mat_<double>(3, 3) << 
    869.1168824543144, 0.0, 480.0,
         0.0, 869.1168824543144, 360.0,
         0.0, 0.0, 1.0);

Mat dist_coeffs = (Mat_<double>(1, 5) << 
    0.0, 0.0, 0.0, 0.0, 0.0
);

// ==========================================
// 模块 2 & 3 的实现（你原本的逻辑）
// ==========================================
bool isImage(const string& path) {
    string ext = path.substr(path.find_last_of(".") + 1);
    for (auto& c : ext) c = tolower(c);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp");
}

void rotationToEuler(const Mat& R, double& roll, double& pitch, double& yaw) {
    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) + R.at<double>(1,0) * R.at<double>(1,0));
    bool singular = sy < 1e-6;
    if (!singular) {
        roll  = atan2(R.at<double>(2,1), R.at<double>(2,2)) * 180.0 / CV_PI;
        pitch = atan2(-R.at<double>(2,0), sy) * 180.0 / CV_PI;
        yaw   = atan2(R.at<double>(1,0), R.at<double>(0,0)) * 180.0 / CV_PI;
    } else {
        roll  = atan2(-R.at<double>(1,2), R.at<double>(1,1)) * 180.0 / CV_PI;
        pitch = atan2(-R.at<double>(2,0), sy) * 180.0 / CV_PI;
        yaw   = 0;
    }
}

// 将欧拉角转为旋转矩阵 (用于反推相机世界系姿态)
Mat eulerToRmat(double roll, double pitch, double yaw) {
    double r = roll * CV_PI / 180.0, p = pitch * CV_PI / 180.0, y = yaw * CV_PI / 180.0;
    Mat Rx = (Mat_<double>(3,3) << 1,0,0, 0,cos(r),-sin(r), 0,sin(r),cos(r));
    Mat Ry = (Mat_<double>(3,3) << cos(p),0,sin(p), 0,1,0, -sin(p),0,cos(p));
    Mat Rz = (Mat_<double>(3,3) << cos(y),-sin(y),0, sin(y),cos(y),0, 0,0,1);
    return Rz * Ry * Rx; 
}

// ==========================================
// 核心：计算法线与世界系位姿
// ==========================================
void computeNormalsAndWorldPose(PoseResult& res) {
    // 1. 提取相机系外法线
    // 根据文档：1,3号板法线为局部Y轴(列1)，2,4号板法线为局部X轴(列0)
    if (res.gt.id == 1 || res.gt.id == 3) {
        res.normal_cam = res.R_cam.col(1).clone(); 
    } else {
        res.normal_cam = res.R_cam.col(0).clone();
    }
    // 归一化
    res.normal_cam /= cv::norm(res.normal_cam);

    // 2. 构建相机在世界系的平移 (文档值)
    Mat t_world_cam = (Mat_<double>(3, 1) << 0.077834, 0.000761, 0.500196);

    // 3. 构建相机在世界系的旋转 R_wc (利用文档 armor_1 反推)
    // R_w_world_armor1 = R_wc * R_cam_armor1  =>  R_wc = R_w_world_armor1 * (R_cam_armor1)^T
    Mat R_wc_1 = eulerToRmat(14.999, 0.158, 38.549);
    Mat R_cam_1 = eulerToRmat(-76.991, -51.243, -178.547);
    Mat R_wc = R_wc_1 * R_cam_1.t();

    // 4. 计算装甲板的世界系位置 t_w_world = R_wc * t_cam + t_world_cam
    Mat t_world_armor = R_wc * (res.tvec / 1000.0) + t_world_cam; // 将mm转为m计算
    
    // 5. 计算装甲板的世界系姿态 R_w_world = R_wc * R_cam
    Mat R_w_world = R_wc * res.R_cam;

    // 6. 计算世界系法线
    res.normal_world = R_wc * res.normal_cam;
    res.normal_world /= cv::norm(res.normal_world);

    // 保存
    res.tvec_world = t_world_armor * 1000.0; // 存回 mm
    res.R_world = R_w_world;
}

PoseResult evaluatePose(const vector<Point2f>& image_points, const vector<Point3f>& obj_points) {
    PoseResult res;
    bool success = solvePnP(obj_points, image_points, camera_matrix, dist_coeffs, res.rvec, res.tvec);
    if (!success) {
        cout << "[警告] PnP 解算失败，跳过此目标。" << endl;
        res.is_valid = false;
        return res;
    }

    res.is_valid = true; // 标记解算成功
    res.sol_x = res.tvec.at<double>(0) / 1000.0;
    res.sol_y = res.tvec.at<double>(1) / 1000.0;
    res.sol_z = res.tvec.at<double>(2) / 1000.0;

    Rodrigues(res.rvec, res.R_cam);
    rotationToEuler(res.R_cam, res.sol_roll, res.sol_pitch, res.sol_yaw);

    // 匹配最近真值
    double min_dist = 1e9;
    for (const auto& gt : gt_list) {
        double d = sqrt(pow(res.sol_x - gt.tx, 2) + pow(res.sol_y - gt.ty, 2) + pow(res.sol_z - gt.tz, 2));
        if (d < min_dist) { min_dist = d; res.gt = gt; }
    }

    res.dx = (res.sol_x - res.gt.tx) * 1000.0;
    res.dy = (res.sol_y - res.gt.ty) * 1000.0;
    res.dz = (res.sol_z - res.gt.tz) * 1000.0;
    res.dist_3d = sqrt(res.dx*res.dx + res.dy*res.dy + res.dz*res.dz);

    // ======= 新增：计算世界系位姿与外法线 =======
    computeNormalsAndWorldPose(res);
    // ============================================

    cout << "--- 匹配装甲板: " << res.gt.desc << " ---" << endl;
    cout << "  [位置] 解算:(" << res.sol_x << "," << res.sol_y << "," << res.sol_z << ")m | 真值:(" << res.gt.tx << "," << res.gt.ty << "," << res.gt.tz << ")m" << endl;
    cout << "  [误差] dx=" << res.dx << "mm, dy=" << res.dy << "mm, dz=" << res.dz << "mm | 3D误差=" << res.dist_3d << "mm" << endl;
    
    // 打印新增的外法线与世界系信息
    cout << "  [相机系法线]: (" << res.normal_cam.at<double>(0) << ", " << res.normal_cam.at<double>(1) << ", " << res.normal_cam.at<double>(2) << ")" << endl;
    cout << "  [世界系位置]: (" << res.tvec_world.at<double>(0) << ", " << res.tvec_world.at<double>(1) << ", " << res.tvec_world.at<double>(2) << ") mm" << endl;
    cout << "  [世界系法线]: (" << res.normal_world.at<double>(0) << ", " << res.normal_world.at<double>(1) << ", " << res.normal_world.at<double>(2) << ")\n" << endl;

    return res;
}

// 在图像上绘制法线向量
void drawNormal(Mat& img, const PoseResult& res) {
    // 将装甲板中心点和沿法线延伸的点(长度100mm)投影回图像
    vector<Point3f> pts_3d = { Point3f(0,0,0), Point3f(res.normal_cam.at<double>(0)*100, res.normal_cam.at<double>(1)*100, res.normal_cam.at<double>(2)*100) };
    vector<Point2f> pts_2d;
    projectPoints(pts_3d, res.rvec, res.tvec, camera_matrix, dist_coeffs, pts_2d);
    
    // 画一条紫色的法线箭头
    cv::arrowedLine(img, pts_2d[0], pts_2d[1], Scalar(255, 0, 255), 2, 8, 0, 0.2);
    putText(img, "Normal", pts_2d[1], FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255), 1);
}

// ==========================================
// 可视化绘制
// ==========================================
void drawYoloResult(Mat& img, const YoloArmor& armor, const PoseResult& res) {
    rectangle(img, armor.bbox, Scalar(0, 255, 0), 1);
    for (int i = 0; i < 4; ++i) {
        circle(img, armor.points[i], 4, Scalar(0, 0, 255), -1);
        line(img, armor.points[i], armor.points[(i+1)%4], Scalar(0, 255, 0), 2);
    }
    drawFrameAxes(img, camera_matrix, dist_coeffs, res.rvec, res.tvec, 50.0f);
    drawNormal(img, res); // 画法线
    string label = "GT:" + to_string(res.gt.id) + " num:" + to_string(armor.class_id);
    putText(img, label, armor.points[0] + Point2f(0, -10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);
}