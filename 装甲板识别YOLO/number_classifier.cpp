#include "number_classifier.hpp"
using namespace cv;
using namespace std;
NumberClassifier::NumberClassifier(const string& model_path, const Size& input_size): input_size_(input_size)
{
    net_ = dnn::readNetFromONNX(model_path);
    net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
}
void NumberClassifier::classify(const Mat& src, vector<YoloArmor>& armors)
{
    for (auto& armor : armors) 
    {
        // 利用 YOLO 输出的 4 个角点做透视变换 (假设顺序: 左上, 左下, 右下, 右上)
        vector<Point2f> src_pts = {
            armor.points[0], // 左上
            armor.points[3], // 右上
            armor.points[2], // 右下
            armor.points[1]  // 左下
        };
        
        vector<Point2f> dst_pts = {
            Point2f(0, 0),
            Point2f(input_size_.width - 1, 0),
            Point2f(input_size_.width - 1, input_size_.height - 1),
            Point2f(0, input_size_.height - 1)
        };

        Mat warp_mat = getPerspectiveTransform(src_pts, dst_pts);
        Mat roi;
        warpPerspective(src, roi, warp_mat, input_size_, INTER_LINEAR, BORDER_REPLICATE);

        Mat gray;
        cvtColor(roi, gray, COLOR_BGR2GRAY);

        // 构造 blob 推理
        Mat blob = dnn::blobFromImage(gray, 1.0 / 255.0, input_size_, Scalar(), false, false, CV_32F);
        net_.setInput(blob);
        Mat prob = net_.forward();

        Point max_loc;
        double max_val;
        minMaxLoc(prob, nullptr, &max_val, nullptr, &max_loc);
        
        // 如果要使用数字结果过滤，可以在这里把低置信度的剔除
        // armor.class_id = max_loc.x; 
    }
}