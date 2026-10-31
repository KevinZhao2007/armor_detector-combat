#include "show.h"           // ★ 必须最先：内含 Eigen + tracker + opencv + using
#include "state_plot.hpp"   // 状态曲线绘制
#include "imu_reader.hpp"   // IMU 日志读取
#include "number_classifier.hpp"

#include <fstream>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv) 
{
    string input_path = (argc > 1) ? argv[1] : "test.avi";   // 默认是 test.avi
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

    // 1. 内参（供重投影与 dist_to_center 使用）
    initCameraIntrinsics(camera_matrix, frame_width, frame_height);
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

        if (kShowLegend) drawLegend(vis_bgr, frame_height);

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
