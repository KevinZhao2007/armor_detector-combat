#include <opencv2/opencv.hpp>
#include "detector.hpp"          
#include "number_classifier.hpp"
#include "show.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    string input_path = (argc > 1) ? argv[1] : "test.avi";
    bool image_mode = isImage(input_path);//默认读取视频，照片需要加路径名称

    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        cerr << "无法打开文件: " << input_path << endl;
        return -1;
    }

    int width  = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps    = cap.get(CAP_PROP_FPS);
    if (fps == 0) fps = 30;

    VideoWriter writer;
    if (!image_mode) {
        writer.open("result_traditional.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height));
    }
    
    NumberClassifier number_clf("mlp.onnx", 0.9f, Size(28, 20)); 
    Detector detector;
    Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        // 1. 检测与分类
        auto armors = detector.detect(frame);
        number_clf.classify(frame, armors);
        
        // 过滤数字识别失败的（保留传统视觉的这个逻辑）
        vector<Armor> valid_armors;
        for (auto& a : armors) {
            if (a.number >= 0) valid_armors.push_back(a);
        }

        Mat vis_bgr = frame.clone();
    if (valid_armors.empty()) {
            putText(vis_bgr, "No Armor Detected", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            cout << "当前帧未检测到装甲板或数字识别失败。" << endl;
        } else {
            for (const auto& a : valid_armors) {
                if (a.corners.size() != 4) continue;

                // 2. 解算与对比
                PoseResult res = evaluatePose(a.corners, obj_points_trad);

                // 处理解算失败的情况
                if (!res.is_valid) {
                    putText(vis_bgr, "PnP Failed", a.center, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                    continue; // 跳过画图
                }

                // 3. 可视化 (只有解算成功才画坐标系)
                drawTraditionalResult(vis_bgr, a, res);
            }
        }


        // 4. 输入输出处理
        if (image_mode) {
            imwrite("result_image_traditional.jpg", vis_bgr);
            cout << "图片处理完成，已保存为 result_image_traditional.jpg" << endl;
            break;
        } else {
            writer.write(vis_bgr);
            imshow("Traditional Detector", vis_bgr);
            if (waitKey(1) == 27) break; 
        }
    }

    cap.release();
    if (!image_mode) writer.release();
    return 0;
}
