#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include "detector.hpp"   // 引用 Armor 结构体
using namespace std;
using namespace cv;
class NumberClassifier
{
public:
    // model_path: mlp.onnx 路径（这里将模型直接放到build目录下）
    // threshold:  置信度阈值，低于此值判为无效
    // input_size: 模型输入尺寸
    NumberClassifier(const string& model_path,
                     float threshold = 0.7f,
                     const Size& input_size = Size(20, 28));

    // 对每个装甲板做数字识别，结果写入 armor.number
 void classify(const Mat& src, vector<Armor>& armors);

private:
    dnn::Net net_;
    float threshold_;
    Size input_size_;

    // 从装甲板 ROI 中裁出数字区域
    Mat extractNumberROI(const Mat& src, const Armor& armor);
};
