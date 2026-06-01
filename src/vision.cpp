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

  float confidence_floor = 0.15F;
  float nms_threshold = 0.45F;
  int input_size = 640;

  Impl(
      const std::string& model_path,
      float confidence_floor_val,
      int input_size_val,
      float nms_threshold_val)
      : confidence_floor(confidence_floor_val),
        nms_threshold(nms_threshold_val),
        input_size(std::max(32, input_size_val)) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
  }
};

YoloVisionBackend::YoloVisionBackend(
    const std::string& model_path,
    float confidence_floor,
    int input_size,
    float nms_threshold)
    : impl_(std::make_unique<Impl>(
          model_path, confidence_floor, input_size, nms_threshold)) {}

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

  const int target_w = impl_->input_size;
  const int target_h = impl_->input_size;
  const float scale = std::min(
      static_cast<float>(target_w) / static_cast<float>(img.cols),
      static_cast<float>(target_h) / static_cast<float>(img.rows));
  const int resized_w = std::max(1, static_cast<int>(std::round(img.cols * scale)));
  const int resized_h = std::max(1, static_cast<int>(std::round(img.rows * scale)));
  const int pad_x = (target_w - resized_w) / 2;
  const int pad_y = (target_h - resized_h) / 2;
  
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resized_w, resized_h));
  cv::Mat letterboxed(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_w, resized_h)));
  cv::cvtColor(letterboxed, letterboxed, cv::COLOR_BGR2RGB);

  std::vector<float> input_tensor_values(1 * 3 * target_w * target_h);
  float* input_data = input_tensor_values.data();

  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < target_h; ++h) {
      for (int w = 0; w < target_w; ++w) {
        input_data[c * target_h * target_w + h * target_w + w] = letterboxed.at<cv::Vec3b>(h, w)[c] / 255.0f;
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
      float class_id = output_data[i * 6 + 5];

      if (static_cast<int>(std::round(class_id)) == person_class_id) {
        static int print_count = 0;
        if (print_count < 20) {
          std::cout << "[Vision] YOLO26 candidate: conf=" << confidence 
                    << ", box=[" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]\n";
          print_count++;
        }
        if (confidence >= impl_->confidence_floor) {
          int left = static_cast<int>(std::round((x1 - pad_x) / scale));
          int top = static_cast<int>(std::round((y1 - pad_y) / scale));
          int right = static_cast<int>(std::round((x2 - pad_x) / scale));
          int bottom = static_cast<int>(std::round((y2 - pad_y) / scale));
          left = std::clamp(left, 0, img.cols);
          top = std::clamp(top, 0, img.rows);
          right = std::clamp(right, 0, img.cols);
          bottom = std::clamp(bottom, 0, img.rows);
          
          int width = right - left;
          int height = bottom - top;
          if (width > 0 && height > 0) {
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
            class_ids.push_back(person_class_id);
          }
        }
      }
    }
  } else if (output_shape.size() == 3) {
    // Standard YOLOv8 output shape: [1, 84, 8400]
    int cols = static_cast<int>(output_shape[2]);
    const int class_offset = 4;

    for (int i = 0; i < cols; ++i) {
      float class_score = output_data[(class_offset + person_class_id) * cols + i];
      if (class_score >= impl_->confidence_floor) {
        float cx = output_data[0 * cols + i];
        float cy = output_data[1 * cols + i];
        float w = output_data[2 * cols + i];
        float h = output_data[3 * cols + i];

        int left = static_cast<int>(std::round((cx - w * 0.5F - pad_x) / scale));
        int top = static_cast<int>(std::round((cy - h * 0.5F - pad_y) / scale));
        int right = static_cast<int>(std::round((cx + w * 0.5F - pad_x) / scale));
        int bottom = static_cast<int>(std::round((cy + h * 0.5F - pad_y) / scale));
        left = std::clamp(left, 0, img.cols);
        top = std::clamp(top, 0, img.rows);
        right = std::clamp(right, 0, img.cols);
        bottom = std::clamp(bottom, 0, img.rows);
        int width = right - left;
        int height = bottom - top;

        if (width > 0 && height > 0) {
          boxes.push_back(cv::Rect(left, top, width, height));
          confidences.push_back(class_score);
          class_ids.push_back(person_class_id);
        }
      }
    }
  }

  std::vector<int> indices;
  if (!boxes.empty()) {
    cv::dnn::NMSBoxes(
        boxes, confidences, impl_->confidence_floor, impl_->nms_threshold, indices);
  }

  int track_id_counter = 0;
  for (int idx : indices) {
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
