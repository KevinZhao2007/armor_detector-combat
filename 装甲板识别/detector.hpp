#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
struct Light : public RotatedRect
{
    Light() = default;
    explicit Light(const vector<Point>& contour): RotatedRect(minAreaRect(contour))
    {
        width  = this->size.width;
        length = this->size.height;
        if (width > length) swap(width, length);

        float angle = this->angle;
        if (angle > 45) angle -= 90;
        if (angle < -45) angle += 90;
        tilt_angle = abs(angle);
    float long_angle = this->angle;
    if (this->size.width < this->size.height) long_angle += 90.0f;
    
    // 转弧度
    float rad = long_angle * CV_PI / 180.0f;
    float dx = std::cos(rad) * length / 2.0f;
    float dy = std::sin(rad) * length / 2.0f;
    
    // 这样算出来的 top 和 bottom 才是真正贴合倾斜灯条边缘的
    top    = center - cv::Point2f(dx, dy);
    bottom = center + cv::Point2f(dx, dy);
    }
    float width = 0, length = 0;
    float tilt_angle = 0;
    Point2f top, bottom;
    int color = 0;  // 0=未知, 1=红, 2=蓝
};

// 装甲板 
struct Armor
{
    Light left, right;
    int color = 0;
    Point2f center;
    vector<Point2f> corners; // 左上, 右上, 右下, 左下
    float width = 0, height = 0;
    int number = -1;
};

// 参数
struct DetectorParams
{
    int   binary_thres       = 70;
    float min_ratio          = 0.05f;//最大最小长宽比例
    float max_ratio          = 0.40f;
    float max_tilt_angle     = 40.0f; //最大倾斜角
    int   color_diff_thresh  = 25;
    float max_light_gap      = 4.0f;
    float max_height_diff    = 0.3f;
};

// 检测器
class Detector
{
public:
    explicit Detector(const DetectorParams& p = DetectorParams()) : params_(p) {}

    vector<Armor> detect(const Mat& rgb_img);  //这里默认读的是RGB模式下的图像

private:
    Mat preprocessImage(const Mat& rgb_img); 
    vector<Light> findLights(const Mat& binary, const Mat& rgb_img); 
    vector<Armor> matchLights(const vector<Light>& lights);

    DetectorParams params_;
};
