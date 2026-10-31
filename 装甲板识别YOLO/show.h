#pragma once
#include <Eigen/Dense>
#include "tracker.hpp"          
#include <opencv2/opencv.hpp>
#include "detector.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace Eigen;

inline constexpr double kImuLag = 0.009;   // 图像时刻 → IMU 时刻：往前 9ms

// 相机系(右/下/前) ≠ 体系(前/左/上)。true = 按 RM 标准做轴映射；
// IMU 与相机同轴安装时改 false。
inline constexpr bool kUseOpenCvToBodyAxes = true;

// 板宽
// 0.0 = 按宽高比自动判断；0.225 / 0.135 = 强制指定。
inline constexpr double kPlateWidthOverride = 0.0;

// ---- 显示开关 ----
inline constexpr bool kShowPlotWindow       = true;   // 实时曲线窗口
inline constexpr bool kDrawOnlyWhenTracking = true;   // 仅 TRACKING 才画预测框
inline constexpr bool kShowLegend           = true;   // 左下角图例

// ---- 调试 ----
inline constexpr int kDebugFrames = 0;                // 打印前 N 帧诊断，0 = 关

// ---- 观测合理性门限 ----
inline constexpr double kMinDist = 0.20;   // 目标最近 20 cm
inline constexpr double kMaxDist = 15.0;   // 最远 15 m
extern Matrix3d    g_R_cam2imu;    // 相机系 → 体系
extern Vector3d    g_t_wc;         // 相机光心在世界系的位置（取 0）
extern double      g_yaw_offset;   // PnP yaw → 模型 yaw 的固定偏移
extern int         g_flip_cnt;     // 180° 翻转次数（诊断）
extern vector<float> g_res_px;     // 每帧残差（像素，诊断）
extern vector<float> g_size_ratio; // 预测/检测 框大小比（诊断）
//  位姿解算与基础绘图
struct PoseResult {
    bool is_valid = false;
    double sol_x = 0.0, sol_y = 0.0, sol_z = 0.0;          // 米
    double sol_roll = 0.0, sol_pitch = 0.0, sol_yaw = 0.0; // 度
    Mat rvec, tvec;                                        // tvec 单位 mm
    Mat R_cam;                                             // 板 → 相机
};

extern vector<Point3f> obj_points_yolo;   // 默认物点（±117.5 mm）
extern Mat camera_matrix;
extern Mat dist_coeffs;

bool isImage(const string& path);
void rotationToEuler(const Mat& R, double& roll, double& pitch, double& yaw);
PoseResult evaluatePose(const vector<Point2f>& image_points, const vector<Point3f>& obj_points);
void drawYoloResult(Mat& img, const YoloArmor& armor, const PoseResult& res);
// 内参初始化：填 fx/fy/cx/cy + 准备投影位姿
void initCameraIntrinsics(const Mat& K, int frame_w, int frame_h);

// 由 4 个角点判断大板(0.225)/小板(0.135)
double plateWidthFromDetection(const YoloArmor& a);

// PnP 结果 → 世界系观测（含 180° 二义性修正与量测协方差）
bool makeObs(const PoseResult& res, const Matrix3d& R_wc, const Vector3d& t_wc, const YoloArmor& armor, ArmorObs& o);

// 求逆，供 cv::projectPoints 使用（每帧一次）
void updateProjectionPose(const Matrix3d& R_wc, const Vector3d& t_wc);

// 世界系点 → 像素（带畸变）
bool projectWorld(const Matrix3d& R_wc, const Vector3d& t_wc,
                  const Vector3d& p_world, Point2f& uv,
                  int img_w = 0, int img_h = 0);

// 画一块板的重投影四边形；返回对角线像素长度（画不出返回 0）
float drawReprojectedArmor(Mat& img, const Vector3d& center_world, double yaw,
                           double w, double h,
                           const Matrix3d& R_wc, const Vector3d& t_wc,
                           const Scalar& color, const char* tag);

// 观测合理性（NaN / 距离 / 高度）
bool obsSane(const ArmorObs& o);

// 右上角状态量
void drawStateText(Mat& img, const ArmorTracker& tr, int frame_w);

// 左下角颜色图例
void drawLegend(Mat& img, int frame_h);

// 中位数（统计）
float medianOf(vector<float> v);
