#include "number_classifier.hpp"

NumberClassifier::NumberClassifier(const std::string& model_path, const cv::Size& input_size)
    : input_size_(input_size)
{
    net_ = cv::dnn::readNetFromONNX(model_path);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

void NumberClassifier::classify(const cv::Mat& src, std::vector<YoloArmor>& armors)
{
    for (auto& armor : armors) {
        // 利用 YOLO 输出的 4 个角点做透视变换 (假设顺序: 左上, 左下, 右下, 右上)
        std::vector<cv::Point2f> src_pts = {
            armor.points[0], // 左上
            armor.points[3], // 右上
            armor.points[2], // 右下
            armor.points[1]  // 左下
        };
        
        std::vector<cv::Point2f> dst_pts = {
            cv::Point2f(0, 0),
            cv::Point2f(input_size_.width - 1, 0),
            cv::Point2f(input_size_.width - 1, input_size_.height - 1),
            cv::Point2f(0, input_size_.height - 1)
        };

        cv::Mat warp_mat = cv::getPerspectiveTransform(src_pts, dst_pts);
        cv::Mat roi;
        cv::warpPerspective(src, roi, warp_mat, input_size_, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

        // 构造 blob 推理
        cv::Mat blob = cv::dnn::blobFromImage(gray, 1.0 / 255.0, input_size_, cv::Scalar(), false, false, CV_32F);
        net_.setInput(blob);
        cv::Mat prob = net_.forward();

        cv::Point max_loc;
        double max_val;
        cv::minMaxLoc(prob, nullptr, &max_val, nullptr, &max_loc);
        
        // 如果要使用数字结果过滤，可以在这里把低置信度的剔除
        // armor.class_id = max_loc.x; 
    }
}