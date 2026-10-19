#pragma once
#include <opencv2/opencv.hpp>
#include "detector.hpp"

class PnpSolver {
public:
    PnpSolver();
    void solve(const std::vector<YoloArmor>& armors, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

private:
    std::vector<cv::Point3f> small_armor_obj_pts;
    std::vector<cv::Point3f> large_armor_obj_pts;
};