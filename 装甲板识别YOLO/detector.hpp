#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
struct YoloArmor {
    cv::Rect bbox;
    std::vector<cv::Point2f> points; // 4个角点 [左上, 左下, 右下, 右上]
    int class_id;                    // 类别 (如装甲板编号)
    int color_id;                    // 0: 蓝, 1: 红
    float confidence;
};

class YoloDetector {
public:
    YoloDetector(const std::string& model_path, float conf_threshold = 0.5f, float nms_threshold = 0.45f);
    std::vector<YoloArmor> detect(const cv::Mat& img);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YoloDetector"};
    Ort::Session session_{nullptr}; // 替换 cv::dnn::Net
    Ort::SessionOptions session_options_;
    std::vector<const char*> input_names_{"images"};  // 模型输入节点名，需用 Netron 确认
    std::vector<const char*> output_names_{"output0"}; // 模型输出节点名，需用 Netron 确认
    float conf_thresh_;
    float nms_thresh_;
    cv::Size input_size_;
};
