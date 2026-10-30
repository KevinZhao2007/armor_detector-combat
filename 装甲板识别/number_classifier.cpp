#include "number_classifier.hpp"
#include <iostream>
using namespace std;
using namespace cv;
NumberClassifier::NumberClassifier(const string& model_path,float threshold,const Size& input_size): threshold_(threshold), input_size_(input_size)
{
    // 初始化ONNX模型
    net_ = dnn::readNetFromONNX(model_path);
    if (net_.empty())
    {
        cerr << "[NumberClassifier] 模型加载失败: " << model_path << endl;
        return;
    }

    // 设置后端和目标设备（CPU）
    net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(dnn::DNN_TARGET_CPU);

    cout << "[NumberClassifier] 模型加载成功: " << model_path << endl;
}

//从装甲板 ROI 中裁出数字区域
Mat NumberClassifier::extractNumberROI(const Mat& src, const Armor& armor)
{
    // 用装甲板中心作为数字区域中心
    // 宽度取两灯条中心距的 60%，高度取灯条平均长度的 60%，强行把照片拉平
    float roi_width  = armor.width * 0.6f;
    float roi_height = armor.height * 0.6f;

    int x = static_cast<int>(armor.center.x - roi_width / 2);
    int y = static_cast<int>(armor.center.y - roi_height / 2);
    int w = static_cast<int>(roi_width);
    int h = static_cast<int>(roi_height);

    // 边界裁剪，防止越界
    x = max(0, min(x, src.cols - 1));
    y = max(0, min(y, src.rows - 1));
    w = min(w, src.cols - x);
    h = min(h, src.rows - y);

    if (w <= 0 || h <= 0) return Mat();
    Rect roi(x, y, w, h);
    return src(roi).clone();
}

// 对装甲板做数字识别
void NumberClassifier::classify(const Mat& src,vector<Armor>& armors)
{
    for (auto& armor : armors)
    {
        // 1. 裁出数字 ROI
        Mat roi = extractNumberROI(src, armor);
        if (roi.empty())
        {
            armor.number = -1;
            continue;
        }
cv::imwrite("debug_roi.jpg", roi); 
        // 2. 预处理：灰度化 + 尺寸缩放 + 归一化
        Mat gray, resized;
        if (roi.channels() == 3)
            cvtColor(roi, gray, COLOR_BGR2GRAY);
        else
            gray = roi;

        resize(gray, resized, input_size_);

        // 3. 构造 blob，送入网络
        // scalefactor = 1/255.0 将像素归一化到 [0,1]
        Mat blob = dnn::blobFromImage(
            resized,
            1.0 / 255.0,          // scalefactor
            input_size_,          // size
            Scalar(),             // mean（MLP 通常不需要减均值）
            false,                // swapRB（灰度图无需交换）
            false,                // crop
            CV_32F                // ddepth
        );

        net_.setInput(blob);
        Mat prob = net_.forward();   // 输出 shape: (1, num_classes)

        // 4. 后处理：取最大概率
        Mat prob_mat = prob.reshape(1, 1);   // 变成 1 x num_classes
        Point max_loc;
        double max_val;
        minMaxLoc(prob_mat, nullptr, &max_val, nullptr, &max_loc);

        // 5. 置信度判断
        if (max_val < threshold_)
            armor.number = -1;   // 置信度太低，判为无效
        else
            armor.number = max_loc.x;   // 类别索引，对应数字编号

        // 调试输出
      // cout << "[NumberClassifier] armor center=("<< armor.center.x << "," << armor.center.y<< ") number=" << armor.number<< " conf=" << max_val << endl;
    }
}
