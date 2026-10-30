#include "detector.hpp"
#include <iostream>
#include <numeric> // for iota
using namespace std;
using namespace cv;
YoloDetector::YoloDetector(const string& model_path, float conf_threshold, float nms_threshold)
    : conf_thresh_(conf_threshold), nms_thresh_(nms_threshold)
{
    // 1. 设置 Session 选项
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. 加载模型
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    // 3. 动态获取输入/输出节点名称 (避免手动填错)
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_.GetInputNameAllocated(0, allocator);
    auto output_name = session_.GetOutputNameAllocated(0, allocator);
    input_names_.push_back(input_name.get());
    output_names_.push_back(output_name.get());

    // 4. 动态获取输入尺寸
    auto input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // YOLO 输入通常是 [1, 3, H, W]
    if (input_shape.size() == 4) {
        input_size_ = Size((int)input_shape[3], (int)input_shape[2]);
    } else {
        cerr << "[YoloDetector] 警告: 无法自动获取输入尺寸，默认使用 640x640" << endl;
        input_size_ = Size(640, 640);
    }
    
    cout << "[YoloDetector] 模型加载成功!" << endl;
    cout << "  输入尺寸: " << input_size_.width << "x" << input_size_.height << endl;
    cout << "  输入节点: " << input_names_[0] << endl;
    cout << "  输出节点: " << output_names_[0] << endl;
}

vector<YoloArmor> YoloDetector::detect(const Mat& img)
{
    vector<YoloArmor> armors;

    // ================= 1. 预处理 =================
    // 如果你需要保持长宽比，请改用 Letterbox，这里使用直接 Resize
    Mat resized;
    resize(img, resized, input_size_);
    
    // 转为 float32，归一化到 [0,1]，并交换 R 和 B 通道 (YOLO 期望 RGB)
    Mat blob = dnn::blobFromImage(resized, 1.0 / 255.0, input_size_, Scalar(), true, false, CV_32F);

    // ================= 2. 准备输入 =================
    vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
    size_t input_tensor_size = blob.total() * blob.elemSize();
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)blob.data, input_tensor_size / sizeof(float),
        input_shape.data(), input_shape.size()
    );

    // ================= 3. 执行推理 =================
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        input_names_.data(), &input_tensor, 1, 
        output_names_.data(), 1
    );

    // ================= 4. 解析输出 =================
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    // YOLOv8 输出形状通常是 [1, 属性数, 预测框数量] 即 [1, 21, 8400]
    if (output_shape.size() != 3) return armors;

    int attr_size = output_shape[1];   // 属性数 (4个bbox + 类别数 + 8个关键点)
    int num_anchors = output_shape[2]; // 预测框数量 (如 8400)
    int num_classes = attr_size - 12; 
    
    // 如果无法推断，保底处理
    if (num_classes <= 0) return armors;

    vector<Rect> boxes;
    vector<float> confidences;
    vector<YoloArmor> temp_armors;

    // 遍历每一个预测框
    for (int i = 0; i < num_anchors; ++i) {
        // 找到类别置信度最大的那一个
        float max_conf = 0.0f;
        int max_cls_id = -1;
        
        // 类别从索引 4 开始，到 4 + num_classes 结束
        for (int c = 0; c < num_classes; ++c) {
            // 注意：数据排布是 [属性][预测框]，所以跨步是 num_anchors
            float conf = output_data[(4 + c) * num_anchors + i]; 
            if (conf > max_conf) {
                max_conf = conf;
                max_cls_id = c;
            }
        }

        if (max_conf > conf_thresh_) {
            // 解析 BBox (cx, cy, w, h)
            float cx = output_data[0 * num_anchors + i];
            float cy = output_data[1 * num_anchors + i];
            float w  = output_data[2 * num_anchors + i];
            float h  = output_data[3 * num_anchors + i];

            // 映射回原图尺寸
            int left   = int((cx - w / 2.0f) * img.cols / input_size_.width);
            int top    = int((cy - h / 2.0f) * img.rows / input_size_.height);
            int width  = int(w * img.cols / input_size_.width);
            int height = int(h * img.rows / input_size_.height);

            // 限制在图像边界内
            left = max(0, min(left, img.cols - 1));
            top = max(0, min(top, img.rows - 1));
            width = min(width, img.cols - left);
            height = min(height, img.rows - top);

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(max_conf);

            YoloArmor armor;
            armor.bbox = boxes.back();
            armor.class_id = max_cls_id;
            armor.confidence = max_conf;
            armor.color_id = 0; // 颜色需要根据模型或像素进一步判断，暂设为 0

            // 解析 4 个关键点 (紧跟在类别之后)
            int kpt_start_idx = 4 + num_classes;
            for (int k = 0; k < 4; ++k) {
                float kp_x = output_data[(kpt_start_idx + k * 2) * num_anchors + i] * img.cols / input_size_.width;
                float kp_y = output_data[(kpt_start_idx + k * 2 + 1) * num_anchors + i] * img.rows / input_size_.height;
                armor.points.emplace_back(kp_x, kp_y);
            }
            temp_armors.push_back(armor);
        }
    }

    // ================= 5. NMS (非极大值抑制) =================
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, indices);
    
    for (int idx : indices) {
        armors.push_back(temp_armors[idx]);
    }

    return armors;
}