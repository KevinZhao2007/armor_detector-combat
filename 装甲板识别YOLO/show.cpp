#include "show.h"
using namespace std;
using namespace cv;
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

