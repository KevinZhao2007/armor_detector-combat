#include "detector.hpp"
#include<iostream>
using namespace std;
using namespace cv;
Mat Detector::preprocessImage(const Mat& bgr_img)
{
    Mat gray, binary;
    cvtColor(bgr_img, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, params_.binary_thres, 255, THRESH_BINARY);
    return binary;
}
// ========== 查找灯条 ==========
vector<Light> Detector::findLights(const Mat& binary, const Mat& bgr)
{
    vector<vector<Point>> contours;
    findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<Light> lights;
    for (const auto& contour : contours)
    {
        if (contour.size() < 5) continue;

        Light light(contour);

        // 形状筛选
        if (light.width > light.length) continue;
        float ratio = light.width / light.length;
        bool ratio_ok = params_.min_ratio < ratio && ratio < params_.max_ratio;
        bool angle_ok = light.tilt_angle < params_.max_tilt_angle;
        if (!ratio_ok || !angle_ok) continue;
        if (light.tilt_angle > params_.max_tilt_angle) continue;
        // 颜色判断：统计轮廓上 R 和 B 通道之和
        int sum_r = 0, sum_b = 0;
        for (const auto& p : contour)
        {
            const auto& px = bgr.at<Vec3b>(p.y, p.x);
            sum_b += px[0];  // B
            sum_r += px[2];  // R
        }
      int contour_size = static_cast<int>(contour.size());
        int avg_r = sum_r / contour_size;
        int avg_b = sum_b / contour_size;
        int diff = avg_r - avg_b;

        // 【修改点 2：过滤不够红/不够蓝的灯条】
        // 如果 R 比 B 大很多，判定为红，但要求 R 的绝对亮度 avg_r 不能太低（比如大于 100）
        if (diff > params_.color_diff_thresh ) 
            light.color = 1; // RED
        // 如果 B 比 R 大很多，判定为蓝，同样要求 B 的绝对亮度 avg_b 不能太低
        else if (-diff > params_.color_diff_thresh) 
            light.color = 2; // BLUE
        else 
            continue; // 颜色纯度不够或太暗，丢弃
        lights.push_back(light);
    }
    return lights;
}

// ========== 灯条配对 ==========
vector<Armor> Detector::matchLights(const vector<Light>& lights)
{
    vector<Armor> armors;
    for (size_t i = 0; i < lights.size(); i++)
    {
        for (size_t j = i + 1; j < lights.size(); j++)
        {
            const Light& l1 = lights[i];
            const Light& l2 = lights[j];

            if (l1.color != l2.color) continue;
            float avg_len_debug = (l1.length + l2.length) / 2;
            float avg_wid_debug = (l1.width + l2.width) / 2;
            float dx_debug = l2.center.x - l1.center.x;
            float dy_debug = l2.center.y - l1.center.y;
            float dist_debug = sqrt(dx_debug*dx_debug + dy_debug*dy_debug);
            
            const Light* left  = &l1;
            const Light* right = &l2;
            if (left->center.x > right->center.x) swap(left, right);

            float avg_len = (left->length + right->length) / 2;
            float avg_wid = (left->width  + right->width)  / 2;
            if (abs(left->length - right->length) / avg_len > params_.max_height_diff)
                continue;

            float dx = right->center.x - left->center.x;
            float dy = right->center.y - left->center.y;
            float dist = sqrt(dx*dx + dy*dy);
            float armor_ratio = dist / avg_len;
        // 正常的装甲板宽高比通常在 1.5 到 4.5 之间
            if (armor_ratio < 1.2f || armor_ratio > 5.0f) continue; 
            if (dist > params_.max_light_gap * avg_len) continue;
            if (dist < avg_wid * 1.5f) continue;
                    float angle_diff = abs(left->tilt_angle - right->tilt_angle);
            float line_angle = atan2(dy, dx) * 180.0f / CV_PI;
            if (abs(line_angle) > 30.0f) continue;  // 连线角度不能太陡
            if (angle_diff > 10.0f) continue;  // 两灯条倾斜角差异不能太大
            Armor armor;
            armor.left  = *left;
            armor.right = *right;
            armor.color = left->color;
            armor.center = (left->center + right->center) / 2;
            armor.width  = dist;
            armor.height = avg_len;

            // 4 个角点：左上、右上、右下、左下
            armor.corners = {
                left->top, right->top,
                right->bottom, left->bottom
            };

            armors.push_back(armor);
        }
    }
    return armors;
}

// ========== 主检测入口 ==========
vector<Armor> Detector::detect(const Mat& bgr_img)
{
    Mat binary = preprocessImage(bgr_img);
    auto lights = findLights(binary, bgr_img);
    auto armors = matchLights(lights);

   // cout << "[Detector] 灯条数=" << lights.size()<< "  装甲板数=" << armors.size() << endl;
    return armors;
}