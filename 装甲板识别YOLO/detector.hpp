#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
using namespace std;
using namespace cv;
struct YoloArmor {
    Rect bbox;
    vector<Point2f> points; // 4个角点 [左上, 左下, 右下, 右上]
    int class_id;                    // 类别 (如装甲板编号)
    int color_id;                    // 0: 蓝, 1: 红
    float confidence;
};

class YoloDetector {
public:
    YoloDetector(const string& model_path, float conf_threshold = 0.5f, float nms_threshold = 0.45f);
    vector<YoloArmor> detect(const Mat& img);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YoloDetector"}; // 新增
    Ort::Session session_{nullptr}; // 替换 dnn::Net
    Ort::SessionOptions session_options_; // 新增
    vector<const char*> input_names_{"images"};  // 模型输入节点名，需用 Netron 确认
    vector<const char*> output_names_{"output0"}; // 模型输出节点名，需用 Netron 确认
    float conf_thresh_;
    float nms_thresh_;
    Size input_size_;
};