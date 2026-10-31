#pragma once
#include <opencv2/opencv.hpp>
#include "detector.hpp"
using namespace std;
using namespace cv;
class PnpSolver {
public:
    PnpSolver();
    void solve(const vector<YoloArmor>& armors, const Mat& camera_matrix, const Mat& dist_coeffs);

private:
    vector<Point3f> small_armor_obj_pts;
    vector<Point3f> large_armor_obj_pts;
};