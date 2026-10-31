#include "show.h"

#include "viz.hpp"      
// 相机系 → 体系 的轴映射（是两套轴定义的固定关系，不是标定值）
static Matrix3d makeCamToImu() {
    if (!kUseOpenCvToBodyAxes) return Matrix3d::Identity();
    Matrix3d R;
    R << 0, 0, 1,
        -1, 0, 0,
         0,-1, 0;
    return R;
}

Matrix3d      g_R_cam2imu = makeCamToImu();
Vector3d      g_t_wc      = Vector3d::Zero();
double        g_yaw_offset = 0.0;
int           g_flip_cnt   = 0;
vector<float> g_res_px;
vector<float> g_size_ratio;

// 投影专用（仅本文件）：世界→相机的 (rvec, tvec) 与主点
static Mat    g_rvec_wc, g_tvec_wc;
static double g_fx = 1.0, g_fy = 1.0, g_cx = 0.0, g_cy = 0.0;

//  中位数（诊断统计）
float medianOf(vector<float> v) {
    if (v.empty()) return 0.f;
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}

//  内参初始化
void initCameraIntrinsics(const Mat& K, int frame_w, int frame_h) {
    g_fx = K.at<double>(0, 0);
    g_fy = K.at<double>(1, 1);
    g_cx = K.at<double>(0, 2);
    g_cy = K.at<double>(1, 2);

    printf("[init] 内参: fx=%.1f fy=%.1f cx=%.1f cy=%.1f（投影用 projectPoints，含畸变）\n",
           g_fx, g_fy, g_cx, g_cy);

    updateProjectionPose(Matrix3d::Identity(), g_t_wc);   // 垫底，之后每帧更新
}

//  左下角图例
void drawLegend(Mat& img, int frame_h) {
    const int y0 = frame_h - 58;
    putText(img, "green : YOLO detection (unchanged)",
            Point(12, y0),      FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 0), 1, LINE_AA);
    putText(img, "yellow: EKF predicted, matched plate",
            Point(12, y0 + 18), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1, LINE_AA);
    putText(img, "pink  : EKF predicted, 3 invisible plates",
            Point(12, y0 + 36), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 140, 255), 1, LINE_AA);
}

vector<Point3f> obj_points_yolo = {
    {-117.5f, -27.5f, 0}, // 左上 (0)
    {-117.5f,  27.5f, 0}, // 左下 (1)
    { 117.5f,  27.5f, 0}, // 右下 (2)
    { 117.5f, -27.5f, 0}  // 右上 (3)
};

Mat camera_matrix = (Mat_<double>(3, 3) << 
    2362.2795952460974, 0, 679.17513921636282, 0, 2336.1916035376717, 529.06858905837635, 0, 0, 1);

Mat dist_coeffs = (Mat_<double>(1, 5) << 
    -0.098561592108101559, 0.11173066531932681, -0.0038352945867108721, -0.0020421443354436304, 0
);

// 模块 2 & 3 的实现（你原本的逻辑）
bool isImage(const string& path) {
    string ext = path.substr(path.find_last_of(".") + 1);
    for (auto& c : ext) c = tolower(c);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp");
}

void rotationToEuler(const Mat& R, double& roll, double& pitch, double& yaw)
{
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
PoseResult evaluatePose(const vector<Point2f>& image_points, const vector<Point3f>& obj_points) {
    PoseResult res;
    bool success = solvePnP(obj_points, image_points, camera_matrix, dist_coeffs, res.rvec, res.tvec);
    if (!success) 
    {
        cout << "[警告] PnP 解算失败，跳过此目标。" << endl;
        res.is_valid = false;
        return res;
    }//处理装甲板解算失败的情况

    res.is_valid = true; // 标记解算成功
    res.sol_x = res.tvec.at<double>(0) / 1000.0;
    res.sol_y = res.tvec.at<double>(1) / 1000.0;
    res.sol_z = res.tvec.at<double>(2) / 1000.0;

    Rodrigues(res.rvec, res.R_cam);
    rotationToEuler(res.R_cam, res.sol_roll, res.sol_pitch, res.sol_yaw);

    return res;
}

void drawYoloResult(Mat& img, const YoloArmor& armor, const PoseResult& res) {
    rectangle(img, armor.bbox, Scalar(0, 255, 0), 1);
    for (int i = 0; i < 4; ++i) {
        circle(img, armor.points[i], 4, Scalar(0, 0, 255), -1);
        line(img, armor.points[i], armor.points[(i+1)%4], Scalar(0, 255, 0), 2);
    }
    drawFrameAxes(img, camera_matrix, dist_coeffs, res.rvec, res.tvec, 50.0f);
    string label = "ID:" + to_string(armor.class_id) + " " + to_string(int(armor.confidence * 100)) + "%";
    putText(img, label, armor.points[0] + Point2f(0, -10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);
}

Matrix3d toEigen3(const Mat& m) {
    Matrix3d M;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M(i, j) = m.at<double>(i, j);
    return M;
}
//  从 4 个角点判断这是大板还是小板（跳变处理）
double plateWidthFromDetection(const YoloArmor& a) 
{
    if (kPlateWidthOverride > 0.0) return kPlateWidthOverride;
    if (a.points.size() != 4) return 0.135;
    const double h = norm(a.points[1] - a.points[0]);   // 左边 = 板高
    const double w = norm(a.points[2] - a.points[1]);   // 下边 = 板宽
    if (h < 1e-6) return 0.135;
    return (w / h > 3.2) ? 0.225 : 0.135;
}
// 把 PnP 结果转成"世界系观测"
//  输入：res.R_cam（板→相机旋转）、res.tvec（板在相机系的位置，mm）
//        R_wc, t_wc（相机→世界，来自 IMU）
//  输出：ArmorObs{z=[xa,ya,za,yaw], R, ...}
bool makeObs(const PoseResult& res,const Matrix3d& R_wc, const Vector3d& t_wc,const YoloArmor& armor, ArmorObs& o)
{
    if (!res.is_valid) return false;

    // 位置：相机系 → 世界系（tvec 是 mm，先换成米）
    const Vector3d p_cam(res.tvec.at<double>(0) / 1000.0,res.tvec.at<double>(1) / 1000.0,res.tvec.at<double>(2) / 1000.0);
    const Vector3d p_world = R_wc * p_cam + t_wc;

    // 板朝向：板自身的法向是局部 z 轴，也就是 R_cam 的第 3 列
    const Matrix3d R_cam = toEigen3(res.R_cam);
    const Vector3d n_world = R_wc * R_cam.col(2);

    // 法向 → 模型的 yaw。模型约定：yaw 从"板"指向"车中心"。
    //    法向从中心指向板，所以先加 π 翻转，再叠加固定偏移。
    double yaw_obs = atan2(n_world(1), n_world(0)) + M_PI + g_yaw_offset;

    //关键修正：消除平面 PnP 的 180° 二义性 
        const Vector2d u(cos(yaw_obs), sin(yaw_obs));
        const Vector2d rel(p_world(0) - t_wc(0), p_world(1) - t_wc(1));
        if (u.dot(rel) < 0.0) 
        {
            yaw_obs += M_PI;                 // 反了 → 翻 180°
            g_flip_cnt++;                    // 仅用于诊断统计
        }
    // 量测协方差：同时按"距离"和"视角"给
    const double d = p_cam.norm();
    // 板在相机系里的法向就是 R_cam 的第 3 列
    const Vector3d n_cam = R_cam.col(2).normalized();
    const double cos_t = max(0.15,fabs(n_cam.dot(p_cam.normalized())));

    const double s_xy  = max(0.010, 0.020 * d);            // 横向 ~2% 距离
    const double s_z   = max(0.020, 0.050 * d) / cos_t;    // ★ 深度：越斜越差
    const double s_yaw = 0.030 / max(0.35, cos_t);         // ★ 朝向：越斜越差

    Matrix4d R_cam4 = Matrix4d::Zero();
    R_cam4.diagonal() << s_xy*s_xy, s_xy*s_xy, s_z*s_z, s_yaw*s_yaw;

    Matrix4d Rw = Matrix4d::Zero();
    Rw.block<3,3>(0,0) = R_wc;
    Rw(3,3) = 1.0;                                     // yaw 是标量，不参与旋转

    o.z = Vector4d(p_world(0), p_world(1), p_world(2), yaw_obs);
    o.R = Rw * R_cam4 * Rw.transpose();
    o.car_id = armor.class_id;                         // 机器号，不是板号
    o.conf   = armor.confidence;
    o.dist_to_center = hypot(armor.points[0].x - g_cx,armor.points[0].y - g_cy);
    return true;
}
//  做法：把 R_wc,t_wc 求逆，得到"世界 → 相机"的 (rvec, tvec)，
//  然后把世界系的点直接当"物点"喂进去：
//      p_cam = R_wcᵀ·(p_w − t_wc) = (R_wcᵀ)·p_w + (−R_wcᵀ·t_wc)
//                                      └ rvec ┘    └── tvec ──┘
void updateProjectionPose(const Matrix3d& R_wc, const Vector3d& t_wc)
{
    const Matrix3d R_cw = R_wc.transpose();
    const Vector3d t_cw = -R_cw * t_wc;
    Mat Rm(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) Rm.at<double>(i, j) = R_cw(i, j);
    Rodrigues(Rm, g_rvec_wc);
    g_tvec_wc = (Mat_<double>(3, 1) << t_cw(0), t_cw(1), t_cw(2));
}

bool projectWorld(const Matrix3d& R_wc, const Vector3d& t_wc,const Vector3d& p_world, Point2f& uv, int img_w, int img_h)
{
    //这里先用 Eigen 判一下，排除点在相机后的情况。
    const Vector3d p_cam = worldToCam(R_wc, t_wc, p_world);
    if (!isfinite(p_cam(0)) || !isfinite(p_cam(2))) return false;
    if (p_cam(2) < 0.2) return false;
    if (p_cam.norm() > 30.0) return false;

    vector<Point3f> obj{ Point3f((float)p_world(0), (float)p_world(1), (float)p_world(2)) };
    vector<Point2f> img;
    projectPoints(obj, g_rvec_wc, g_tvec_wc, camera_matrix, dist_coeffs, img);
    if (img.empty()) return false;
    uv = img[0];

    //像素坐标也要合理：偏出画面几倍就说明状态炸了，不画
    if (!isfinite(uv.x) || !isfinite(uv.y)) return false;
    if (img_w > 0 && (abs(uv.x) > 3 * img_w || abs(uv.y) > 3 * img_h))
        return false;
    return true;
}

// 返回投影后四边形的"对角线长度"（像素）；画不出来返回 0。
// ★ 用它自动判断"板宽假设对不对"：预测框/检测框 的比值应 ≈ 1.0。
float drawReprojectedArmor(Mat& img, const Vector3d& center_world, double yaw, double w, double h, const Matrix3d& R_wc, const Vector3d& t_wc, const Scalar& color, const char* tag)
{
    // 板宽沿水平切向 t = [-sin yaw, cos yaw]，板高沿 z
    const Vector3d t (-sin(yaw), cos(yaw), 0.0);
    const Vector3d up(0.0, 0.0, h * 0.5);
    const Vector3d hw = t * (w * 0.5);

    Vector3d c3[4] = { center_world - hw + up,   // 左上
                              center_world - hw - up,   // 左下
                              center_world + hw - up,   // 右下
                              center_world + hw + up }; // 右上
    Point2f uv[4];
    for (int i = 0; i < 4; ++i)
        if (!projectWorld(R_wc, t_wc, c3[i], uv[i], img.cols, img.rows)) return 0.f;

    for (int i = 0; i < 4; ++i) line(img, uv[i], uv[(i + 1) % 4], color, 2);
    if (tag) putText(img, tag, uv[0] + Point2f(0, -6),FONT_HERSHEY_SIMPLEX, 0.45, color, 1, LINE_AA);

    return max((float)norm(uv[0] - uv[2]), (float)norm(uv[1] - uv[3]));
}
// 观测合理性检查
bool obsSane(const ArmorObs& o) 
{
    for (int i = 0; i < 4; ++i)
        if (!isfinite(o.z(i))) return false;
    const double d = hypot(o.z(0), o.z(1));
    if (d < kMinDist || d > kMaxDist) return false;
    if (o.z(2) < -1.0 || o.z(2) > 3.0) return false;
    return true;
}
//  右上角显示实时状态量
void drawStateText(Mat& img, const ArmorTracker& tr, int frame_w)
{
    const char* kSt[4] = {"LOST", "DETECT", "TRACK", "TEMP_LOST"};
    char l1[128], l2[128], l3[128];
    if (!tr.alive()) 
    {
        snprintf(l1, sizeof(l1), "state: LOST");
        snprintf(l2, sizeof(l2), " ");
        snprintf(l3, sizeof(l3), " ");
    } else 
    {
        const auto& x = tr.x();
        snprintf(l1, sizeof(l1), "%s  plate=%d  NIS=%.1f",
                 kSt[int(tr.state())], tr.plate(), tr.nis());
        snprintf(l2, sizeof(l2), "xc=%6.2f  yc=%6.2f  za=%.2f", x(0), x(2), x(4));
        snprintf(l3, sizeof(l3), "vx=%5.2f  vy=%5.2f  yaw=%6.1f  w=%5.2f",
                 x(1), x(3), x(6) * 180.0 / M_PI, x(7));
    }

    const char* ls[3] = {l1, l2, l3};
    int maxw = 0;
    for (int i = 0; i < 3; ++i)
        maxw = max(maxw, getTextSize(ls[i], FONT_HERSHEY_SIMPLEX, 0.55, 2, nullptr).width);

    // 半透明底板，保证白背景上也看得清。
    // 这里所有坐标都做了 clamp —— 图像尺寸变化时不会因为 Rect 越界而抛异常。
    const int boxH = 78;
    int x0 = frame_w - maxw - 20;
    if (x0 < 4) x0 = 4;
    int rx = x0 - 6; if (rx < 0) rx = 0;
    int ry = 8;      if (ry < 0) ry = 0;
    int rw = maxw + 18;
    if (rx + rw > img.cols) rw = img.cols - rx;
    int rh = boxH;
    if (ry + rh > img.rows) rh = img.rows - ry;

    if (rw > 0 && rh > 0) {
        Mat roi = img(Rect(rx, ry, rw, rh));
        roi.convertTo(roi, -1, 0.55, 0);      // 原地变暗 45%，比 addWeighted 简单
    }

    for (int i = 0; i < 3; ++i)
        putText(img, ls[i], Point(x0, 30 + i * 24),
                FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 255), 2, LINE_AA);
}
