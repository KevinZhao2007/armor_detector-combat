#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "detector.hpp" // 如果传统视觉，确保 Armor 定义在这里

using namespace std;
using namespace cv;

struct GroundTruth {
    int id;
    double tx, ty, tz;
    double roll, pitch, yaw; 
    string desc;
};

struct PoseResult {
    bool is_valid = false;
    GroundTruth gt;
    // 相机系
    double sol_x, sol_y, sol_z;
    double dx, dy, dz, dist_3d;
    double sol_roll, sol_pitch, sol_yaw;
    Mat rvec, tvec;
    Mat R_cam; // 相机系旋转矩阵
    Mat normal_cam; // 相机系外法线

    // 世界系 (odom_vision)
    Mat tvec_world;
    Mat R_world;
    Mat normal_world;
};

// 全局变量声明
extern vector<GroundTruth> gt_list;
extern vector<Point3f> obj_points_yolo;   // 传统视觉用 obj_points_trad
extern Mat camera_matrix;
extern Mat dist_coeffs;

// 函数声明
bool isImage(const string& path);
void rotationToEuler(const Mat& R, double& roll, double& pitch, double& yaw);
PoseResult evaluatePose(const vector<Point2f>& image_points, const vector<Point3f>& obj_points);

// 新增：世界系转换与法线计算
void computeNormalsAndWorldPose(PoseResult& res);
void drawNormal(Mat& img, const PoseResult& res);

// 绘制函数
void drawYoloResult(Mat& img, const YoloArmor& armor, const PoseResult& res);