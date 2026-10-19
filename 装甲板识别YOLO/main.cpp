#include <opencv2/opencv.hpp>
#include "detector.hpp" // 或者是你的 yolo_detector.hpp，请保持原样
#include "number_classifier.hpp"
#include "show.h"
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
using namespace cv;
int main(int argc, char** argv) {
    string input_path = (argc > 1) ? argv[1] : "test.avi";
    bool image_mode = isImage(input_path);

    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        cerr << "无法打开文件: " << input_path << endl;
        return -1;
    }

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    if (fps == 0) fps = 30;

    VideoWriter writer;
    if (!image_mode) {
        writer.open("result_yolo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frame_width, frame_height));
    }

    // 初始化 YOLO 模型 (请确保路径正确)
    YoloDetector yolo("Infantry-v8n-fp16-20260726-D1.8w-B16.onnx", 0.5f, 0.45f);
    NumberClassifier num_clf("mlp.onnx", Size(28, 20)); // 如果 YOLO 端到端输出类别，此模块可选

    Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        // 1. 检测
        auto armors = yolo.detect(frame);
        num_clf.classify(frame, armors); // 如果不用 MLP，可注释

        Mat vis_bgr = frame.clone();
        if (armors.empty()) {
            putText(vis_bgr, "No Armor Detected", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            cout << "当前帧未检测到装甲板。" << endl;
        } else {
            for (const auto& armor : armors) {
                if (armor.points.size() != 4) continue;

                // 2. 解算与对比
                PoseResult res = evaluatePose(armor.points, obj_points_yolo);

                // 【新增：安检2】处理解算失败的情况
                if (!res.is_valid) {
                    putText(vis_bgr, "PnP Failed", armor.points[0], FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                    continue; // 跳过画图，直接看下一个装甲板
                }

                // 3. 可视化 (只有解算成功才画坐标系)
                drawYoloResult(vis_bgr, armor, res);
            }
        }

        // 4. 输入输出处理 (保持不变)
        if (image_mode) {
            imwrite("result_image_yolo.jpg", vis_bgr);
            break;
        } else {
            writer.write(vis_bgr);
            imshow("YOLO Detector", vis_bgr);
            if (waitKey(1) == 27) break; 
        }
    }
    cap.release();
    if (!image_mode) writer.release();
    return 0;
}