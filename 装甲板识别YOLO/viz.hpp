// ============================================================================
//  viz.hpp —— 重投影用的纯几何工具（只依赖 Eigen）
//  【本文件是新增的，但只保留"几何"部分】
//
//  为什么还需要这个文件？
//    投影本身不再自己算 —— 直接用 OpenCV 的 cv::projectPoints（见 main.cpp），
//    它会带上畸变，和你原来 drawFrameAxes 用的是同一套。
//
//    这里只留下"把一块板变成 4 个角点"这一步：
//    它是一块矩形在三维空间里的摆放问题，和相机模型无关，
//    所以放在这里，方便单独测试。
// ============================================================================
#pragma once
#include <Eigen/Dense>
#include <cmath>

// 世界 → 相机。只用来判断"点在不在相机后面"（projectPoints 不做这个检查）
inline Eigen::Vector3d worldToCam(const Eigen::Matrix3d& R_wc, const Eigen::Vector3d& t_wc,
                                  const Eigen::Vector3d& p_world) {
    return R_wc.transpose() * (p_world - t_wc);
}

// 一块板的 4 个角点（世界系）
//   板宽沿水平切向 t = [−sin yaw, cos yaw, 0]，板高沿 z
//   注意：yaw 加 180° 得到的是同一组角点，所以 ±π 的二义性不影响画框
inline void plateCorners(const Eigen::Vector3d& center, double yaw,
                         double w, double h, Eigen::Vector3d out[4]) {
    const Eigen::Vector3d t (-std::sin(yaw), std::cos(yaw), 0.0);
    const Eigen::Vector3d up(0.0, 0.0, h * 0.5);
    const Eigen::Vector3d hw = t * (w * 0.5);
    out[0] = center - hw + up;   // 左上
    out[1] = center - hw - up;   // 左下
    out[2] = center + hw - up;   // 右下
    out[3] = center + hw + up;   // 右上
}
