#include "tracker.hpp"
#include <algorithm>
#include <cstdio>
static constexpr double kPi = 3.14159265358979323846;
using namespace std;
using namespace Eigen;
//  把任意角度折回 (-π, π]
//    状态里的 yaw 必须连续增长（比如从 0 长到 12 rad），
//    否则 wyaw 积分会出错，而且换圈时会凭空跳一次。
static inline double wrapPi(double a) {
    a = fmod(a + kPi, 2.0 * kPi);
    if (a < 0.0) a += 2.0 * kPi;
    return a - kPi;
}
//  初始协方差
//    xc/yc : 0.3m       vx/vy : 3m/s
//    za    : 0.2m       vz    : 1m/s
//    yaw   : 0.5rad     wyaw  : 4rad/s
//  速度给得比较松（σ=3），是因为初始化时完全不知道目标速度。
TrackerConfig::TrackerConfig() {
    P0diag << 0.09, 9.0, 0.09, 9.0, 0.04, 1.0, 0.25, 16.0;
}

//  观测方程：第 j 块板应该被"看到"什么样
//    a = yaw + phi[j]        该板的朝向
//    xa = xc - r*cos(a)      从中心往板的方向走 r 米（注意负号，见下）
//    ya = yc - r*sin(a)
//    za = za + dz[j]
//  符号说明：本实现约定 yaw 指向"从板到车中心"。设板在中心沿 a 方向
//  半径 r 处，则 板 = 中心 - r*[cos a, sin a]
Vector4d ArmorTracker::hfun(const VectorXd& x, int j) const 
{
    const double a = x(6) + cfg_.phi[j];
    const double r = cfg_.r[j];
    return Vector4d{
        x(0) - r * cos(a),      // xa
        x(2) - r * sin(a),      // ya
        x(4) + cfg_.dz[j],      // za
        a                       // 板的 yaw
    };
}

//  观测雅可比 H = ∂h/∂x    (4 行 × 8 列)
//  第 i 行 = 第 i 个量测分量，第 j 列 = 第 j 个状态。
//  对 a 求导时 a = yaw + phi，所以 ∂/∂yaw 就等于 ∂/∂a：
//      ∂xa/∂a = +r·sin(a)        ∂ya/∂a = -r·cos(a)
//  其余偏导要么是 1（直接对应），要么是 0（不参与）。
Matrix<double,4,8> ArmorTracker::Hfun(const VectorXd& x, int j) const {
    const double a = x(6) + cfg_.phi[j];
    const double r = cfg_.r[j];

    Matrix<double,4,8> H = Matrix<double,4,8>::Zero();
    //        xc     vx     yc     vy     za     vz     yaw            wyaw
    H(0,0) = 1.0;                                        H(0,6) =  r * sin(a);
    H(1,2) = 1.0;                                        H(1,6) = -r * cos(a);
    H(2,4) = 1.0;
    H(3,6) = 1.0;
    return H;
}

//  过程噪声 Q(dt)
//  连续白噪声加速度（CWNA）对每一对 (位置 i, 速度 v) 的离散化结果：
//      Q(i,i) = σa²·dt⁴/4      位置方差
//      Q(i,v) = σa²·dt³/2      位置-速度交叉项 
//      Q(v,v) = σa²·dt²        速度方差
static void addCWNA(MatrixXd& Q, int i, int v, double dt, double sa) 
{
    const double s2 = sa * sa;
    const double t2 = dt * dt, t3 = t2 * dt, t4 = t3 * dt;
    Q(i, i) += s2 * t4 / 4.0;
    Q(i, v) += s2 * t3 / 2.0;
    Q(v, i) += s2 * t3 / 2.0;
    Q(v, v) += s2 * t2;
}

MatrixXd ArmorTracker::makeQ(double dt) const 
{
    MatrixXd Q = MatrixXd::Zero(8, 8);
    addCWNA(Q, 0, 1, dt, cfg_.a_xy);   // xc, vx
    addCWNA(Q, 2, 3, dt, cfg_.a_xy);   // yc, vy
    addCWNA(Q, 4, 5, dt, cfg_.a_z);    // za, vz
    addCWNA(Q, 6, 7, dt, cfg_.a_w);    // yaw, wyaw
    return Q;
}

//  预测：x⁻ = F·x ，P⁻ = F·P·Fᵀ + Q
void ArmorTracker::predict(double dt) 
{
    MatrixXd F = MatrixXd::Identity(8, 8);
    F(0,1) = F(2,3) = F(4,5) = F(6,7) = dt;   // 四个"位置 += 速度×dt"

    x_ = F * x_;                                  // 线性模型，直接乘 F
    P_ = F * P_ * F.transpose() + makeQ(dt);
}

//  初始化：把"看到的第一块板"定义为 0 号板
//  这里必须要求
void ArmorTracker::initFrom(const ArmorObs& o) {
    const int    j = 0;
    const double a = o.z(3);        // 观测到的板朝向 = 0 号板朝向
    const double r = cfg_.r[j];

    x_.setZero();
    // 观测方程的逆运算：中心 = 板位置 + r·[cos, sin]
    x_(0) = o.z(0) + r * cos(a);
    x_(2) = o.z(1) + r * sin(a);
    x_(4) = o.z(2) - cfg_.dz[j];
    x_(6) = a;
    // 速度全部从 0 起步

    P_ = cfg_.P0diag.asDiagonal();
    plate_  = 0;
    car_id_ = o.car_id;
}

//  匹配：枚举"哪条观测 × 哪块板"，取马氏距离最小的一对
//
//  这一步同时完成四件事：
//    ① 选观测        —— 多条检测里选一条
//    ② 定板号 j      —— 不需要靠 yaw 跳变去猜
//    ③ 剔除离群点    —— 马氏距离 > chi2 门限就拒
//    ④ 输出 NIS      —— 用来验收协方差准不准（均值应 ≈ 4）
//
//  注意：car_id 是"几号机器人"，不是"第几块板"。板号由这里算出来。
bool ArmorTracker::associate(const vector<ArmorObs>& obs,int& k_best, int& j_best, double& nis_best) const
{
    k_best = -1; j_best = -1; nis_best = 1e18;

    for (int k = 0; k < static_cast<int>(obs.size()); ++k) {
        // 只跟同一辆车；car_id < 0 表示未知，就不过滤
        if (car_id_ >= 0 && obs[k].car_id != car_id_) continue;

        for (int j = 0; j < 4; ++j) {
            Vector4d e = obs[k].z - hfun(x_, j);
            e(3) = wrapPi(e(3));                 // ★ 角度残差必须折回

            Matrix<double,4,8> H = Hfun(x_, j);
            Matrix4d S = H * P_ * H.transpose() + obs[k].R;

            // 马氏距离平方 = 归一化新息平方 NIS
            const double nis = e.transpose() * S.ldlt().solve(e);

            if (nis < nis_best) { nis_best = nis; k_best = k; j_best = j; }
        }
    }
    return nis_best < cfg_.chi2_gate;
}
//  更新：x⁺ = x⁻ + K·e ，P⁺ 用 Joseph 形式
//  为什么用 Joseph 形式而不是 (I-KH)P？
//    (I-KH)P 在浮点运算下会慢慢丢掉"对称正定"这个性质，
//    跑几十秒后 P 可能出现负方差。Joseph 形式在数学上等价，
//    但数值上始终保持对称正定：
//        P⁺ = (I-KH)·P⁻·(I-KH)ᵀ + K·R·Kᵀ
void ArmorTracker::updateWith(const ArmorObs& o, int j) {
    Vector4d e = o.z - hfun(x_, j);
    e(3) = wrapPi(e(3));

    Matrix<double,4,8> H = Hfun(x_, j);
    Matrix4d S = H * P_ * H.transpose() + o.R;

    // K = P·Hᵀ·S⁻¹ ；用 ldlt().solve(I) 求 S⁻¹，比 .inverse() 稳
    Matrix4d I4 = Matrix4d::Identity();
    MatrixXd K  = P_ * H.transpose() * S.ldlt().solve(I4);

    x_ += K * e;

    MatrixXd IKH = MatrixXd::Identity(8, 8) - K * H;
    P_ = IKH * P_ * IKH.transpose() + K * o.R * K.transpose();
}

//  主循环：预测 → 匹配 → 更新 → 状态机
bool ArmorTracker::update(const vector<ArmorObs>& obs, double stamp) {
    // ---- 还没初始化：挑"离图像中心最近"的一个观测启动 ----
    if (state_ == TrackState::LOST) {
        const ArmorObs* best = nullptr;
        for (const auto& o : obs)
            if (!best || o.dist_to_center < best->dist_to_center) best = &o;
        if (!best) return false;

        initFrom(*best);
        state_ = TrackState::DETECTING;
        detect_cnt_ = 0; lost_cnt_ = 0;
        last_stamp_ = stamp;
        nis_ = -1.0;
        return false;                     // 本帧只初始化，不做更新
    }

    // ---- 预测 ----
    double dt = stamp - last_stamp_;
    if (!(dt > 0.0)) dt = 1e-3;           // 时间戳重复/倒退时的兜底
    predict(dt);
    last_stamp_ = stamp;

    // 丢失阈值按**时间**折算成帧数（帧率变化时"5 帧"含义会变）
    lost_thres_ = max(1, static_cast<int>(lround(cfg_.lost_time / dt)));

    // ---- 匹配 ----
    int k = -1, j = -1; double nis = 1e18;
    const bool matched = associate(obs, k, j, nis);

    // ---- 更新（不匹配就只保留预测，即 TEMP_LOST 行为）----
    if (matched) {
        updateWith(obs[k], j);
        plate_ = j;
        nis_ = nis;
        nis_sum_ += nis; nis_cnt_ += 1.0;
    } else 
        nis_ = -1.0;
    //   DETECTING : 刚初始化，连续匹配 detect_thres 帧才算稳住
    //   TRACKING  : 正常工作
    //   TEMP_LOST : 短暂丢失，只预测不更新，超过 lost_thres 帧才重置
    switch (state_) 
    {
        case TrackState::DETECTING:
            if (matched) 
            {
                 lost_cnt_ = 0;
                if (++detect_cnt_ >= cfg_.detect_thres) 
                {
                    detect_cnt_ = 0; lost_cnt_ = 0;
                    state_ = TrackState::TRACKING;
                }
            } else 
            {
                if (++lost_cnt_ > 3) 
                {
                    detect_cnt_ = 0; lost_cnt_ = 0;
                    state_ = TrackState::LOST;
                }      // 还没稳住就丢了 → 重新来
            }
            break;
        case TrackState::TRACKING:
            if (!matched) 
            { 
                state_ = TrackState::TEMP_LOST;
                 lost_cnt_ = 1; 
            }
            break;
        case TrackState::TEMP_LOST:
            if (matched) { state_ = TrackState::TRACKING; lost_cnt_ = 0; }
            else if (++lost_cnt_ > lost_thres_) { reset(); }
            break;

        case TrackState::LOST:
            break;
    }
    return matched;
}
//  重置。注意 nis_sum_/nis_cnt_ 不清零 —— 那是整段视频的统计量
void ArmorTracker::reset() {
    state_ = TrackState::LOST;
    x_ = VectorXd::Zero(8);
    P_ = MatrixXd::Identity(8, 8) * 100.0;
    last_stamp_ = 0.0;
    plate_ = -1; car_id_ = -1; nis_ = -1.0;
    detect_cnt_ = 0; lost_cnt_ = 0; lost_thres_ = 10;
}

//  几何查询（画重投影用）
Vector3d ArmorTracker::armorPos(int j) const 
{
    const double a = x_(6) + cfg_.phi[j];
    return Vector3d{ x_(0) - cfg_.r[j] * cos(a),
                            x_(2) - cfg_.r[j] * sin(a),
                            x_(4) + cfg_.dz[j] };
}

Vector3d ArmorTracker::armorPosAt(int j, double dt) const 
{
    // 用当前速度匀速外推。真正按下发时刻预测时，dt = 图像年龄+指令延迟+飞行时间
    const double cx = x_(0) + x_(1) * dt;
    const double cy = x_(2) + x_(3) * dt;
    const double za = x_(4) + x_(5) * dt;
    const double a  = x_(6) + x_(7) * dt + cfg_.phi[j];
    return Vector3d{ cx - cfg_.r[j] * cos(a),
                            cy - cfg_.r[j] * sin(a),
                            za + cfg_.dz[j] };
}

Vector3d ArmorTracker::centerAt(double dt) const 
{
    return Vector3d{ x_(0) + x_(1) * dt,x_(2) + x_(3) * dt,x_(4) + x_(5) * dt };
}

//  状态合理性检查
//  这里的阈值都放得很宽，只拦"明显不可能"的值，不影响正常跟踪。
bool ArmorTracker::sane() const 
{
    for (int i = 0; i < 8; ++i) {
        if (!isfinite(x_(i))) return false;
        if (!isfinite(P_(i, i)) || P_(i, i) < 0.0) return false;
    }
    const double dist = hypot(x_(0), x_(2));      // 车中心到世界原点
    if (dist > 30.0) return false;                     // 场地不会这么大
    if (abs(x_(4)) > 5.0) return false;           // 高度离谱
    if (hypot(x_(1), x_(3)) > 40.0) return false; // 速度离谱
    if (abs(x_(7)) > 40.0) return false;          // 角速度离谱
    return true;
}
//  自检：解析 H 与中心差分对比
//    数值差分：(h(x+εe_j) - h(x-εe_j)) / (2ε)
//    正常情况下返回值应在 1e-9 量级。不是的话就是某个符号写反了。
double ArmorTracker::checkH(int j) const 
{
    const Matrix<double,4,8> Ha = Hfun(x_, j);
    const double eps = 1e-6;
    double worst = 0.0;
    for (int c = 0; c < 8; ++c) {
        VectorXd xp = x_, xm = x_;
        xp(c) += eps; xm(c) -= eps;
        Vector4d num = (hfun(xp, j) - hfun(xm, j)) / (2.0 * eps);
        for (int r = 0; r < 4; ++r)
            worst = max(worst, abs(num(r) - Ha(r, c)));
    }
    return worst;
}
