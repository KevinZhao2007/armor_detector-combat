#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>
using namespace std;
using namespace Eigen;
//  一帧里的一块装甲板观测（已在世界系）
struct ArmorObs {
    Vector4d z;                    // [xa, ya, za, yaw]，米 / 弧度
    Matrix4d R;                    // 量测协方差（随距离变化）
    int   car_id = -1;                    // YOLO 给的机器人编号（**不是板号**）
    float conf = 0.0f;
    float dist_to_center = 1e9f;          // 距图像中心，仅用于初始化时挑一个
};

//  四块板的几何参数
//  下标 j 就是"板号"，0 号板为参考板。
//  注意：对角两块板同类 → r[0]=r[2]、r[1]=r[3]、w[0]=w[2]、w[1]=w[3]
struct TrackerConfig 
{
    array<double,4> phi {{ 0.0, M_PI/2, M_PI, 3*M_PI/2 }};  // 相对 0 号板的方位
    array<double,4> r   {{ 0.20, 0.20, 0.20, 0.20 }};        // 板到车中心 (m)
    array<double,4> dz  {{ 0.0,  0.0,  0.0,  0.0 }};         // 相对 0 号板的高度差
    array<double,4> w   {{ 0.135,0.135,0.135,0.135 }};       // 板宽（小/大交替）
    double h_armor = 0.055;                                       // 板高 (m)

    // PnP 的 yaw → 模型 yaw 的固定偏移。
    // 模型约定 yaw 从"板"指向"车中心"，而 PnP 的法向通常指向外侧，
    // 两者差 π；再加上平面 PnP 本身的 180° 二义性，统一用这个常数吸收。
    // 标定方法：静止目标，比较打印出来的 yaw_obs 与 tracker 的 yaw，
    //           差 π 就把这里设成 M_PI。
    double yaw_offset = 0.0;

    // 过程噪声：各方向"猜不到的加速度"的标准差
    double a_xy = 6.0;    // m/s²   （步兵横移加速度量级）
    double a_z  = 1.5;    // m/s²
    double a_w  = 4.0;    // rad/s²（自旋角加速度）

    double chi2_gate    = 13.28;  // chi2(4 自由度) 99% 分位
    int    detect_thres = 5;      // 连续匹配多少帧才进入 TRACKING
    double lost_time    = 0.3;    // 超过这么久没观测就重置 (s)

    Matrix<double,8,1> P0diag;   // 初始协方差对角

    TrackerConfig();
};

// 跟踪状态机（和开源实现同名，方便对照）
enum class TrackState { LOST, DETECTING, TRACKING, TEMP_LOST };

//  跟踪器
class ArmorTracker {
public:
    ArmorTracker() { reset(); }

    TrackerConfig&       cfg()       { return cfg_; }
    const TrackerConfig& cfg() const { return cfg_; }

    void reset();

    // 主入口：喂一帧的所有观测。
    // 返回 true 表示本帧真的做了 EKF 更新（false = 只预测 / 刚初始化 / 未匹配）
    bool update(const vector<ArmorObs>& obs, double stamp);

    // ---- 给外部查询 ----
    TrackState state()      const { return state_; }
    bool  alive()           const { return state_ != TrackState::LOST; }
    bool  tracking()        const { return state_ == TrackState::TRACKING; }
    const VectorXd& x() const { return x_; }
    const MatrixXd& P() const { return P_; }
    int    plate()          const { return plate_; }    // 本帧匹配到的板号
    double nis()            const { return nis_; }      // 本帧 NIS，-1 表示未更新
    double meanNis()        const { return nis_cnt_ > 0 ? nis_sum_/nis_cnt_ : 0.0; }
    int    carId()          const { return car_id_; }

    Vector3d armorPos(int j) const;              // 第 j 块板当前位置
    Vector3d armorPosAt(int j, double dt) const; // 未来 dt 秒后的板位置
    Vector3d centerAt(double dt) const;          // 未来 dt 秒后的车中心

    // 自检：用中心差分验证解析 Jacobian。第一次跑就打印它，应 < 1e-6
    double checkH(int j = 0) const;

    // 状态合理性检查。
    //   发散时（NaN、速度离谱、位置离谱）返回 false，
    //   调用方据此**不要画重投影**，避免画出横穿屏幕的巨型框。
    bool sane() const;

    // 暴露出来，方便画四块板的重投影
    Vector4d hfun(const VectorXd& x, int j) const;
    Matrix<double,4,8> Hfun(const VectorXd& x, int j) const;

private:
    void initFrom(const ArmorObs& o);
    bool associate(const vector<ArmorObs>& obs,
                   int& k_best, int& j_best, double& nis_best) const;
    MatrixXd makeQ(double dt) const;
    void predict(double dt);
    void updateWith(const ArmorObs& o, int j);

    TrackerConfig cfg_;

    TrackState state_ = TrackState::LOST;
    VectorXd x_ = VectorXd::Zero(8);
    MatrixXd P_ = MatrixXd::Identity(8, 8);

    double last_stamp_ = 0.0;
    int    plate_   = -1;
    int    car_id_  = -1;
    double nis_     = -1.0;
    double nis_sum_ = 0.0, nis_cnt_ = 0.0;

    int detect_cnt_ = 0;
    int lost_cnt_   = 0;
    int lost_thres_ = 10;
};
