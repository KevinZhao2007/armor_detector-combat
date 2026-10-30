#include "tracker.hpp"        // 整车状态 EKF+ Eigen
#include <opencv2/opencv.hpp>
#include "detector.hpp"
#include "number_classifier.hpp"
#include "show.h"
#include "imu_reader.hpp"     // 你已有的 IMU 读取
#include "state_plot.hpp"     // 状态曲线绘制（新增）
#include "viz.hpp"            // 重投影工具（新增）
#include <bits/stdc++.h>         // 方便用 STL 容器、算法、I/O 流等
using namespace std;
using namespace cv;
using namespace Eigen;
constexpr double kImuLag = 0.009;   // 图像时刻 → IMU 时刻：往前 9ms
static Mat g_rvec_wc;                  // 世界 → 相机 的旋转（Rodrigues 向量）
static Mat g_tvec_wc;                  // 世界 → 相机 的平移
static double g_fx = 1.0, g_fy = 1.0;  // 只用来算 dist_to_center 用的主点
static double g_cx = 0.0, g_cy = 0.0;
static const bool kUseOpenCvToBodyAxes = true;
static Matrix3d makeCamToImu()
{
    if (!kUseOpenCvToBodyAxes) return Matrix3d::Identity();
    Matrix3d R;
    R << 0, 0, 1,
        -1, 0, 0,
         0,-1, 0;
    return R;
}
static Matrix3d g_R_cam2imu = makeCamToImu();

// 相机光心在世界系的位置。目标距离几米时，几厘米的偏心可以忽略，
// 所以先设 0（等价于"世界系原点定在相机光心"）。
static Vector3d g_t_wc = Vector3d::Zero();

// PnP 的 yaw → 模型 yaw 的固定偏移。
// ★ 现在 180° 二义性已经由 makeObs() 里的几何判据自动处理，
//   这个常量只在"板的局部坐标系另有约定"时才需要非零。
static double g_yaw_offset = 0.0;

// 诊断：有多少次观测触发了 180° 翻转（说明 PnP 确实在跳分支）
static int g_flip_cnt = 0;

// 诊断：每帧的残差（像素）与"预测框/检测框"大小比
static vector<float> g_res_px;
static vector<float> g_size_ratio;

static float medianOf(vector<float> v) 
{
    if (v.empty()) return 0.f;
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}
// 是否显示实时的曲线窗口（跑视频时建议 true，跑批处理时 false）
static const bool kShowPlotWindow = true;
// ★ 板宽。0 = 按"宽高比"自动判断（斜视时会判错，见下方说明）；
//   0.225 = 强制大板，0.135 = 强制小板。
static const double kPlateWidthOverride = 0.0;

// 只在 TRACKING 状态才画预测框（DETECTING 阶段状态还没收敛，画出来会乱跳）
static const bool kDrawOnlyWhenTracking = true;

// 是否在左下角显示图例
static const bool kShowLegend = true;

// 打印前 N 帧的诊断信息（观察量是否合理、IMU 姿态对不对）
static const int  kDebugFrames = 0;

// 观测合理性阈值：明显不可能的量测直接丢掉，不让它污染滤波器
static const double kMinDist = 0.20;   // 目标最近 20cm
static const double kMaxDist = 15.0;   // 最远 15m
//  【★新增】小工具：Mat <-> Eigen
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

bool projectWorld(const Matrix3d& R_wc, const Vector3d& t_wc,const Vector3d& p_world, Point2f& uv, int img_w = 0, int img_h = 0)
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
int main(int argc, char** argv) 
{
    string input_path = (argc > 1) ? argv[1] : "test.avi";   // 默认仍是 test.avi
    bool image_mode = isImage(input_path);
    //输入类型的判别
    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        cerr << "无法打开文件: " << input_path << endl;
        return -1;
    }

    int frame_width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(CAP_PROP_FPS);
    if (fps == 0) fps = 30;

    VideoWriter writer;
    if (!image_mode) {
        writer.open("result_yolo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    fps, Size(frame_width, frame_height));
    }

    YoloDetector yolo("Infantry-v8n-fp16-20260726-D1.8w-B16.onnx", 0.5f, 0.45f);
    NumberClassifier num_clf("mlp.onnx", Size(28, 20));

    // 1. 内参 → g_K（供重投影使用）
    
    // 投影统一交给 projectPoints，这里只需要主点（算 dist_to_center）
    g_fx = camera_matrix.at<double>(0, 0);
    g_fy = camera_matrix.at<double>(1, 1);
    g_cx = camera_matrix.at<double>(0, 2);
    g_cy = camera_matrix.at<double>(1, 2);
    printf("[main] 内参: fx=%.1f fy=%.1f cx=%.1f cy=%.1f（投影用 projectPoints，含畸变）\n",
           g_fx, g_fy, g_cx, g_cy);

    updateProjectionPose(Matrix3d::Identity(), g_t_wc);   // 垫底，之后每帧更新
    // 2. 载入 IMU
    ImuLog imu;
    const bool imu_ok = imu.load("imu.txt");
    if (!imu_ok) 
    {
        printf("\n\n");
        printf("  错误: 没读到 imu.txt\n");
        printf("  - 确认格式是:  t(s)  qw  qx  qy  qz\n");
        printf("  没有 IMU 姿态时，世界系会跟着云台一起转，\n");
        printf("  目标看起来在【绕圈】，EKF 永远进不了 TRACKING。\n");
        printf("\n\n");
    } else 
        printf("[main] IMU: %zu 条, t = %.3f ~ %.3f s\n",imu.s.size(), imu.s.front().t, imu.s.back().t);

    // 3. 初始化整车状态估计器
    ArmorTracker tracker;
    tracker.cfg().r   = {{0.20, 0.20, 0.20, 0.20}};      // 板到车中心半径
    tracker.cfg().phi = {{0.0, M_PI/2, M_PI, 3*M_PI/2}}; // 四块板方位
    tracker.cfg().w   = {{0.135, 0.135, 0.135, 0.135}};  // 小/大板交替
    if (kPlateWidthOverride > 0.0)
        tracker.cfg().w = {{kPlateWidthOverride, kPlateWidthOverride,
                            kPlateWidthOverride, kPlateWidthOverride}};
    tracker.cfg().yaw_offset = 0.0;                       // ← 差 180° 时改成 M_PI
    g_yaw_offset = tracker.cfg().yaw_offset;

    // 自检：解析 Jacobian 与数值差分对比，正常应 < 1e-6
    // ★ 内参自检：主点应该 ≈ 画面中心
    printf("[main] checkH = %.2e （应 < 1e-6；否则是 H 的符号写错了）\n",
           tracker.checkH(0));
    // 4. 状态记录（曲线 + CSV）
    StateHistory hist;
    ofstream csv("state_log.csv");
    csv << "t,xc,vx,yc,vy,za,vz,yaw,wyaw,"
           "s_xc,s_vx,s_yc,s_vy,s_za,s_vz,s_yaw,s_wyaw,state,plate,nis\n";
    csv << setprecision(6);  
    Mat frame;
    while (cap.read(frame)) 
    {
        if (frame.empty()) break;
        const double t_img = cap.get(CAP_PROP_POS_MSEC) / 1000.0;
        Matrix3d R_wc = Matrix3d::Identity();
        Vector3d t_wc = g_t_wc;

        if (imu_ok)
        {
            // ★ 关键：取图像时间戳"之前 9ms"的姿态（内录 IMU 相对图像滞后）
            const array<double,4> q = imu.at(t_img - kImuLag);
            const auto Rm = ImuLog::toRot(q);        // array<array<3>,3>
            Matrix3d R_imu;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) R_imu(i, j) = Rm[i][j];
            R_wc = R_imu * g_R_cam2imu;              // IMU → 相机
        }
        // 把 R_wc,t_wc 求逆，准备给 projectPoints 用
        updateProjectionPose(R_wc, t_wc);
        // 1. 检测（原样保留）
        auto armors = yolo.detect(frame);
        num_clf.classify(frame, armors);

        Mat vis_bgr = frame.clone();
        vector<ArmorObs> obs;
        vector<Point2f> det_centers;         // ★ 检测到的板的像素中心（画残差用）
        vector<float>   det_diag;            // ★ 检测框对角线长度（判板宽用）
        vector<double>  det_w;  
        if (armors.empty()) 
        {
            // 原样保留：没检测到就提示，跟踪器会自动进入 TEMP_LOST（只预测）
            putText(vis_bgr, "No Armor Detected", Point(50, 50),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        for (const auto& armor : armors) 
        {
            if (armor.points.size() != 4) continue;
            //  改为：按检测到的大小选板宽
            const double w_plate = plateWidthFromDetection(armor);
            vector<Point3f> obj = {
                Point3f((float)(-w_plate * 500.0f), -27.5f, 0),
                Point3f((float)(-w_plate * 500.0f),  27.5f, 0),
                Point3f((float)( w_plate * 500.0f),  27.5f, 0),
                Point3f((float)( w_plate * 500.0f), -27.5f, 0)
            };
            PoseResult res = evaluatePose(armor.points, obj);
            if (!res.is_valid) {
                putText(vis_bgr, "PnP Failed", armor.points[0],
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                continue;
            }

            // 3. 可视化（原样保留：检测框 + 角点 + 坐标轴）
            drawYoloResult(vis_bgr, armor, res);

            // ★ 记录这块板的像素中心，后面用来量"预测框偏了多少像素"
            Point2f pc(0.f, 0.f);
            for (const auto& q : armor.points) pc += q;
            pc *= 0.25f;
            det_centers.push_back(pc);
            float dg = 0.f;
            for (int i = 0; i < 4; ++i)
                dg = max(dg, (float)norm(armor.points[i] - armor.points[(i + 2) % 4]));
            det_diag.push_back(dg);
            det_w.push_back(w_plate); 
            // ★ 新增：转成世界系观测
            ArmorObs o;
            if (makeObs(res, R_wc, t_wc, armor, o) && obsSane(o)) {
                // ★ 软启动：还在 DETECTING 时把量测方差放大，
                //   让滤波器慢慢吸收，避免第一帧的坏深度把状态一把带偏。
                if (tracker.state() == TrackState::DETECTING) o.R *= 2.0;
                obs.push_back(o);
            } else if (kDebugFrames > 0) {
                printf("[skip] 观测不合理: z=(%.2f,%.2f,%.2f) yaw=%.2f\n",
                       o.z(0), o.z(1), o.z(2), o.z(3));
            }
        }
        // 5. 更新整车状态（预测 + 匹配 + 更新）
        //  即使本帧没有观测，也要调用：内部会只做预测（TEMP_LOST）
        tracker.update(obs, t_img);
        // 6. 四块板重投影
        //  注意：这里画的是"状态预测出的 4 块板"，不是检测到的板。
        //  如果 4 块（包括当前看不到的那 3 块）都贴住图像上的板，
        //  就说明状态、观测方程、坐标变换全都对了。
        const bool pred_ok = tracker.alive() && tracker.sane()&& (!kDrawOnlyWhenTracking || tracker.tracking());
        if (pred_ok) 
        {
               for (int j = 0; j < 4; ++j) 
               {
                const bool hit = (j == tracker.plate());
                if (!hit) continue;
                // 把"离预测板最近的检测框"找出来
                int     best_k = -1;
                float   best_d = 1e9f;
                Point2f pp(0.f, 0.f);
                if (hit && !det_centers.empty() &&
                    projectWorld(R_wc, t_wc, tracker.armorPos(j), pp,
                                 frame_width, frame_height)) {
                    for (int k = 0; k < (int)det_centers.size(); ++k) {
                        const float dd = (float)norm(det_centers[k] - pp);
                        if (dd < best_d) { best_d = dd; best_k = k; }
                    }
                }
                // ★画框宽度
                const double draw_w =
                    (best_k >= 0 && best_k < (int)det_w.size())
                        ? det_w[best_k]
                        : tracker.cfg().w[j];

                const Scalar col = hit ? Scalar(0, 255, 255)      // 匹配到的板：黄
                                       : Scalar(200, 140, 255);   // 其余三块：粉
                const float pred_diag = drawReprojectedArmor(
                                     vis_bgr,
                                     tracker.armorPos(j),
                                     tracker.x()(6) + tracker.cfg().phi[j],
                                     draw_w, tracker.cfg().h_armor,
                                     R_wc, t_wc, col, hit ? "hit" : nullptr);

                // ---- 残差线 + 大小比 ----
                if (best_k >= 0) {
                    line(vis_bgr, pp, det_centers[best_k], Scalar(0, 0, 255), 1, LINE_AA);
                    char rb[64];
                    snprintf(rb, sizeof(rb), "%.0fpx", best_d);
                    putText(vis_bgr, rb, (pp + det_centers[best_k]) * 0.5f,
                            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 1, LINE_AA);

                    g_res_px.push_back(best_d);
                    if (pred_diag > 1.f && det_diag[best_k] > 1.f)
                        g_size_ratio.push_back(pred_diag / det_diag[best_k]);
                }
            }
            // 车中心（绿点）
            Point2f c;
            if (projectWorld(R_wc, t_wc, tracker.centerAt(0), c, frame_width, frame_height))
                circle(vis_bgr, c, 4, Scalar(0, 255, 0), -1);
        }

        // 7. 右上角状态量 + 左下角图例
        drawStateText(vis_bgr, tracker, frame_width);

        if (kShowLegend) {
            //  图例：解释三种颜色分别是什么，避免把"预测的看不见的板"当成画错
            const int y0 = frame_height - 58;
            putText(vis_bgr, "green : YOLO detection (unchanged)",
                    Point(12, y0),      FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 0), 1, LINE_AA);
            putText(vis_bgr, "yellow: EKF predicted, matched plate",
                    Point(12, y0 + 18), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1, LINE_AA);
            putText(vis_bgr, "pink  : EKF predicted, 3 invisible plates",
                    Point(12, y0 + 36), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 140, 255), 1, LINE_AA);
        }

        //8. 记录 + 实时曲线窗口
        if (tracker.alive()) {
            hist.push(t_img, tracker.x(), tracker.P(), int(tracker.state()));
            const auto& x = tracker.x();
            const auto& P = tracker.P();
            csv << t_img;
            for (int i = 0; i < 8; ++i) csv << "," << x(i);
            for (int i = 0; i < 8; ++i) csv << "," << sqrt(max(0.0, P(i,i)));
            csv << "," << int(tracker.state()) << "," << tracker.plate()
                << "," << tracker.nis() << "\n";
        }
        if (kShowPlotWindow && !hist.t.empty()) {
            Mat panel = drawStatePanel(hist, 760, 600, 10.0);
            imshow("states", panel);
        }
        // 4. 输入输出处理（原样保留）
        if (image_mode) 
        {
            imwrite("result_image_yolo.jpg", vis_bgr);
            break;
        }
        else 
        {
            writer.write(vis_bgr);
            imshow("YOLO Detector", vis_bgr);
            if (waitKey(1) == 27) break;
        }
    }
    // 9. 收尾：把整段曲线存成 PNG（作业要的那张图）
    if (!hist.t.empty())
    {
        Mat full = drawStatePanel(hist, 1520, 1140, -1.0);   // -1 = 全部历史
        imwrite("state_curves.png", full);
        printf("[main] 已保存 state_curves.png (%zu 帧)\n", hist.t.size());
        printf("[main] NIS 均值 = %.2f （4 维量测，理想 ≈ 4）\n", tracker.meanNis());
        printf("[main] 180° 翻转次数 = %d\n", g_flip_cnt);
        printf("[main] 残差中位数 = %.1f px      （目标 < 30）\n", medianOf(g_res_px));
        printf("[main] 预测/检测 框大小比中位数 = %.2f （目标 ≈ 1.00；\n"
               "       若稳定在 1.67 或 0.60，说明板宽假设错了）\n", medianOf(g_size_ratio));
    }
    csv.close();
    cap.release();
    if (!image_mode) writer.release();
    return 0;
}
