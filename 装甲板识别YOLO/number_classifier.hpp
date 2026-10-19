#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "detector.hpp"
using namespace std;
class NumberClassifier {
public:
    NumberClassifier(const std::string& model_path, const cv::Size& input_size = cv::Size(28, 20));
    void classify(const cv::Mat& src, std::vector<YoloArmor>& armors);

private:
    cv::dnn::Net net_;
    cv::Size input_size_;
};