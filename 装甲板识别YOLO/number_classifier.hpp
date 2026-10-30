#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "detector.hpp"
using namespace std;
using namespace cv;
class NumberClassifier {
public:
    NumberClassifier(const string& model_path, const Size& input_size = Size(28, 20));
    void classify(const Mat& src, vector<YoloArmor>& armors);

private:
    dnn::Net net_;
    Size input_size_;
};