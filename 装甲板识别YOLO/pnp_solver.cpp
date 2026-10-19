#include "pnp_solver.hpp"

PnpSolver::PnpSolver() {
    // 小装甲板 (宽 135mm, 高 55mm)
    float w = 135.0f / 2, h = 55.0f / 2;
    // 顺序：左上, 左下, 右下, 右上
    small_armor_obj_pts = {
        {-w, -h, 0}, // 左上
        {-w,  h, 0}, // 左下
        { w,  h, 0}, // 右下
        { w, -h, 0}  // 右上
    };
    // 大装甲板
    float wl = 225.0f / 2, hl = 55.0f / 2;
    large_armor_obj_pts = {
        {-wl, -hl, 0}, {-wl, hl, 0}, {wl, hl, 0}, {wl, -hl, 0}
    };
}

void PnpSolver::solve(const std::vector<YoloArmor>& armors, const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) {
    for (const auto& armor : armors) {
        if (armor.points.size() != 4) continue;

        // 根据尺寸选择 (假设类别 ID 无法区分大小，实际工程中可能需要判断 bbox 宽高)
        std::vector<cv::Point3f> obj_pts = small_armor_obj_pts; 

        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(obj_pts, armor.points, camera_matrix, dist_coeffs, rvec, tvec);
        
        if (success) {
            std::cout << "PnP 成功! Z距离 = " << tvec.at<double>(2) << " mm" << std::endl;
        }
    }
}