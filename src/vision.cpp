#include "speaker_id/modules/vision.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace speaker_id {

struct YoloVisionBackend::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "YoloVision"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;

  float confidence_threshold = 0.35f;

  Impl(const std::string& model_path, float threshold_val)
      : confidence_threshold(threshold_val) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
  }
};

YoloVisionBackend::YoloVisionBackend(const std::string& model_path, float confidence_threshold)
    : impl_(std::make_unique<Impl>(model_path, confidence_threshold)) {}

YoloVisionBackend::~YoloVisionBackend() = default;

std::vector<TrackEvent> YoloVisionBackend::AcceptFrame(const FrameEvent& frame) {
  std::vector<TrackEvent> tracks;
  if (frame.data.empty()) {
    return tracks;
  }

  cv::Mat img;
  if (frame.data.size() > 4 && frame.data[0] == 0xFF && frame.data[1] == 0xD8) {
    img = cv::imdecode(frame.data, cv::IMREAD_COLOR);
  } else {
    if (frame.width > 0 && frame.height > 0 && frame.data.size() == static_cast<size_t>(frame.width * frame.height * 3)) {
      img = cv::Mat(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.data.data())).clone();
    }
  }

  if (img.empty()) {
    return tracks;
  }

  const int target_w = 640;
  const int target_h = 640;
  
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(target_w, target_h));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  std::vector<float> input_tensor_values(1 * 3 * target_w * target_h);
  float* input_data = input_tensor_values.data();

  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < target_h; ++h) {
      for (int w = 0; w < target_w; ++w) {
        input_data[c * target_h * target_w + h * target_w + w] = resized.at<cv::Vec3b>(h, w)[c] / 255.0f;
      }
    }
  }

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> input_shape = {1, 3, target_h, target_w};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

  const char* input_names[] = {"images"};
  const char* output_names[] = {"output0"};
  Ort::Value inputs[] = {std::move(input_tensor)};

  auto output_tensors = impl_->session->Run(
      Ort::RunOptions{nullptr},
      input_names,
      inputs,
      1,
      output_names,
      1
  );

  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  
  static bool shape_printed = false;
  if (!shape_printed) {
    std::cout << "[Vision] YOLO output shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
      std::cout << (i > 0 ? ", " : "") << output_shape[i];
    }
    std::cout << "]\n";
    shape_printed = true;
  }
  
  const int person_class_id = 0;
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  if (output_shape.size() == 3 && output_shape[1] == 300 && output_shape[2] == 6) {
    // YOLO26 End-to-End (NMS-free) output shape: [1, 300, 6]
    int num_detections = static_cast<int>(output_shape[1]);
    for (int i = 0; i < num_detections; ++i) {
      float x1 = output_data[i * 6 + 0];
      float y1 = output_data[i * 6 + 1];
      float x2 = output_data[i * 6 + 2];
      float y2 = output_data[i * 6 + 3];
      float confidence = output_data[i * 6 + 4];
      float class_id_val = output_data[i * 6 + 5];
      int class_id = static_cast<int>(std::round(class_id_val));

      if (class_id == person_class_id) {
        static int print_count = 0;
        if (print_count < 20) {
          std::cout << "[Vision] YOLO26 candidate: conf=" << confidence 
                    << ", box=[" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]\n";
          print_count++;
        }
      }

      if (confidence >= impl_->confidence_threshold) {
        int left = static_cast<int>(std::max(0.0f, x1) / target_w * img.cols);
        int top = static_cast<int>(std::max(0.0f, y1) / target_h * img.rows);
        int right = static_cast<int>(std::min(static_cast<float>(target_w), x2) / target_w * img.cols);
        int bottom = static_cast<int>(std::min(static_cast<float>(target_h), y2) / target_h * img.rows);
        
        int width = right - left;
        int height = bottom - top;
        if (width > 0 && height > 0) {
          boxes.push_back(cv::Rect(left, top, width, height));
          confidences.push_back(confidence);
          class_ids.push_back(class_id);
        }
      }
    }
  } else if (output_shape.size() == 3) {
    // Standard YOLOv8 output shape: [1, 84, 8400]
    int cols = static_cast<int>(output_shape[2]);
    int num_classes = static_cast<int>(output_shape[1]) - 4;
    const int class_offset = 4;

    for (int i = 0; i < cols; ++i) {
      int best_class_id = -1;
      float max_class_score = 0.0f;
      for (int c = 0; c < num_classes; ++c) {
        float score = output_data[(class_offset + c) * cols + i];
        if (score > max_class_score) {
          max_class_score = score;
          best_class_id = c;
        }
      }

      if (best_class_id >= 0 && max_class_score >= impl_->confidence_threshold) {
        float cx = output_data[0 * cols + i];
        float cy = output_data[1 * cols + i];
        float w = output_data[2 * cols + i];
        float h = output_data[3 * cols + i];

        int left = static_cast<int>((cx - w * 0.5f) / target_w * img.cols);
        int top = static_cast<int>((cy - h * 0.5f) / target_h * img.rows);
        int width = static_cast<int>(w / target_w * img.cols);
        int height = static_cast<int>(h / target_h * img.rows);

        boxes.push_back(cv::Rect(left, top, width, height));
        confidences.push_back(max_class_score);
        class_ids.push_back(best_class_id);
      }
    }
  }

  std::vector<int> indices;
  if (!boxes.empty()) {
    cv::dnn::NMSBoxes(boxes, confidences, impl_->confidence_threshold, 0.45f, indices);
  }

  int track_id_counter = 0;
  for (int idx : indices) {
    if (class_ids[idx] != person_class_id) {
      continue;
    }
    TrackEvent track;
    track.person_track_id = ++track_id_counter;
    track.bbox = BBox{
      static_cast<float>(boxes[idx].x),
      static_cast<float>(boxes[idx].y),
      static_cast<float>(boxes[idx].x + boxes[idx].width),
      static_cast<float>(boxes[idx].y + boxes[idx].height)
    };
    track.quality = confidences[idx] > 0.6f ? Quality::kGood : Quality::kOk;
    track.confidence = confidences[idx];
    tracks.push_back(track);
  }

  return tracks;
}

} // namespace speaker_id
