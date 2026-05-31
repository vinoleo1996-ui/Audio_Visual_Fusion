#include "speaker_id/modules/face.hpp"
#include "speaker_id/modules/kalman_tracker.hpp"

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <chrono>
#include <unordered_set>

namespace speaker_id {

// L2-normalize helper
std::vector<float> NormalizeEmbedding(const std::vector<float>& vec) {
  double norm_sq = 0.0;
  for (float v : vec) norm_sq += static_cast<double>(v) * v;
  double norm = std::sqrt(norm_sq);
  
  std::vector<float> out(vec.size());
  if (norm > 1e-6) {
    for (size_t i = 0; i < vec.size(); ++i) {
      out[i] = static_cast<float>(vec[i] / norm);
    }
  } else {
    out = vec;
  }
  return out;
}

// Similarity score
float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0F;
  double dot = 0.0;
  for (size_t i = 0; i < a.size(); ++i) dot += static_cast<double>(a[i]) * b[i];
  return static_cast<float>(dot);
}

// BBox matching helper (IoU)
float FacePersonMatchScore(const BBox& face, const BBox& person) {
  float face_cx = (face.x1 + face.x2) * 0.5F;
  float face_cy = (face.y1 + face.y2) * 0.5F;
  float person_h = person.y2 - person.y1;
  bool inside = (face_cx >= person.x1 && face_cx <= person.x2 &&
                 face_cy >= person.y1 && face_cy <= person.y2);
  bool in_head_y = (face_cy >= person.y1 && face_cy <= person.y1 + person_h * 0.45F);
  
  float xA = std::max(face.x1, person.x1);
  float yA = std::max(face.y1, person.y1);
  float xB = std::min(face.x2, person.x2);
  float yB = std::min(face.y2, person.y2);
  float inter = std::max(0.0F, xB - xA) * std::max(0.0F, yB - yA);
  
  float faceArea = (face.x2 - face.x1) * (face.y2 - face.y1);
  float personArea = (person.x2 - person.x1) * (person.y2 - person.y1);
  float unionArea = faceArea + personArea - inter;
  float overlap = (unionArea > 0.0F) ? (inter / unionArea) : 0.0F;
  
  float score = overlap;
  if (inside) score += 0.15F;
  if (in_head_y) score += 0.25F;
  return score;
}

// ROI generation logic
cv::Rect GetHeadROI(const BBox& bbox, int width, int height) {
  float p_w = std::max(1.0F, bbox.x2 - bbox.x1);
  float p_h = std::max(1.0F, bbox.y2 - bbox.y1);
  float pad_x = p_w * 0.12F;
  float pad_top = p_h * 0.06F;
  int roi_x1 = std::max(0, static_cast<int>(std::round(bbox.x1 - pad_x)));
  int roi_y1 = std::max(0, static_cast<int>(std::round(bbox.y1 - pad_top)));
  int roi_x2 = std::min(width, static_cast<int>(std::round(bbox.x2 + pad_x)));
  int roi_y2 = std::min(height, static_cast<int>(std::round(bbox.y1 + p_h * 0.68F)));
  
  if (roi_x2 <= roi_x1 || roi_y2 <= roi_y1) {
    return cv::Rect();
  }
  return cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1);
}

// BBox dynamic projection fallback
BBox DefaultProjectFaceBBox(const BBox& body_bbox) {
  float p_w = body_bbox.x2 - body_bbox.x1;
  float p_h = body_bbox.y2 - body_bbox.y1;
  return BBox{
    body_bbox.x1 + p_w * 0.15F,
    body_bbox.y1 + p_h * 0.05F,
    body_bbox.x2 - p_w * 0.15F,
    body_bbox.y1 + p_h * 0.35F
  };
}

// BBox expansion for full head/face coverage in rendering
BBox ExpandBBox(const BBox& box, float scale_x, float scale_y, int img_width, int img_height) {
  float w = box.x2 - box.x1;
  float h = box.y2 - box.y1;
  float dx = w * (scale_x - 1.0F) * 0.5F;
  float dy = h * (scale_y - 1.0F) * 0.5F;
  return BBox{
    std::max(0.0F, box.x1 - dx),
    std::max(0.0F, box.y1 - dy * 1.5F), // Shift upwards to cover forehead/hair
    std::min(static_cast<float>(img_width), box.x2 + dx),
    std::min(static_cast<float>(img_height), box.y2 + dy * 0.8F) // Less down to avoid chest
  };
}

// Laplacian Variance Blur Score
double ComputeBlurScore(const cv::Mat& gray) {
  if (gray.empty() || gray.rows < 2 || gray.cols < 2) return 0.0;
  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  return stddev[0] * stddev[0];
}

double ComputeFaceBlurScore(const cv::Mat& frame, const BBox& bbox) {
  const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(bbox.x1))));
  const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(bbox.y1))));
  const int x2 = std::max(0, std::min(frame.cols, static_cast<int>(std::round(bbox.x2))));
  const int y2 = std::max(0, std::min(frame.rows, static_cast<int>(std::round(bbox.y2))));
  if (x2 <= x1 || y2 <= y1) return 0.0;
  cv::Mat gray;
  cv::cvtColor(frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)), gray, cv::COLOR_BGR2GRAY);
  return ComputeBlurScore(gray);
}

std::vector<std::pair<float, float>> ProjectLandmarks(
    const std::vector<std::pair<float, float>>& landmarks,
    const BBox& source_bbox,
    const BBox& target_bbox) {
  const float source_w = std::max(1.0F, source_bbox.x2 - source_bbox.x1);
  const float source_h = std::max(1.0F, source_bbox.y2 - source_bbox.y1);
  const float target_w = std::max(1.0F, target_bbox.x2 - target_bbox.x1);
  const float target_h = std::max(1.0F, target_bbox.y2 - target_bbox.y1);
  std::vector<std::pair<float, float>> projected;
  projected.reserve(landmarks.size());
  for (const auto& landmark : landmarks) {
    const float rel_x = (landmark.first - source_bbox.x1) / source_w;
    const float rel_y = (landmark.second - source_bbox.y1) / source_h;
    projected.push_back({
        target_bbox.x1 + rel_x * target_w,
        target_bbox.y1 + rel_y * target_h,
    });
  }
  return projected;
}

// Anchor helper
std::vector<std::pair<float, float>> GenerateAnchors(int width, int height, int stride, int num_anchors) {
  int h = height / stride;
  int w = width / stride;
  std::vector<std::pair<float, float>> centers;
  centers.reserve(h * w * num_anchors);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float cx = static_cast<float>(x * stride);
      float cy = static_cast<float>(y * stride);
      for (int k = 0; k < num_anchors; ++k) {
        centers.push_back({cx, cy});
      }
    }
  }
  return centers;
}

struct FaceEngine::Impl {
  // buffalo_l detector exports static shape hints even though its graph accepts
  // smaller dynamic inputs. Suppress those known metadata warnings in live logs.
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "FaceEngine"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> det_session;
  std::unique_ptr<Ort::Session> rec_session;

  Impl(const std::string& model_dir, [[maybe_unused]] bool use_cuda) {
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifndef __APPLE__
    if (use_cuda) {
      // CUDA Execution Provider if requested/available
      auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
      (void)status;
    }
#endif

    std::string det_path = model_dir + "/det_10g.onnx";
    std::string rec_path = model_dir + "/w600k_r50.onnx";

    det_session = std::make_unique<Ort::Session>(env, det_path.c_str(), session_options);
    rec_session = std::make_unique<Ort::Session>(env, rec_path.c_str(), session_options);
  }
};

FaceEngine::FaceEngine(const FaceConfig& config, const std::string& model_dir, bool use_cuda)
    : config_(config), model_dir_(model_dir), use_cuda_(use_cuda) {
  impl_ = std::make_unique<Impl>(model_dir, use_cuda);
  ReloadGallery();
}

FaceEngine::~FaceEngine() = default;

void FaceEngine::ReloadGallery(const std::string& gallery_path) {
  std::lock_guard<std::mutex> lock(face_mu_);
  last_gallery_path_ = gallery_path;
  
  std::ifstream in(gallery_path);
  if (!in) {
    std::cout << "[FaceEngine] No existing gallery found at " << gallery_path << ", dynamic matching only.\n";
    static_gallery_.clear();
    return;
  }

  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  static_gallery_.clear();

  size_t pos = 0;
  while (true) {
    size_t key_pos = content.find('"', pos);
    if (key_pos == std::string::npos) break;
    size_t key_end = content.find('"', key_pos + 1);
    if (key_end == std::string::npos) break;
    
    std::string key = content.substr(key_pos + 1, key_end - key_pos - 1);
    pos = key_end + 1;
    
    bool is_numeric = !key.empty() && std::all_of(key.begin(), key.end(), ::isdigit);
    if (is_numeric) {
      int id = std::stoi(key);
      
      size_t obj_start = content.find('{', pos);
      size_t obj_end = content.find('}', obj_start + 1);
      if (obj_start == std::string::npos || obj_end == std::string::npos) break;
      
      std::string obj_str = content.substr(obj_start, obj_end - obj_start + 1);
      pos = obj_end + 1;
      
      GalleryItem item;
      
      size_t name_pos = obj_str.find("\"name\"");
      if (name_pos != std::string::npos) {
        size_t val_start = obj_str.find('"', name_pos + 6);
        size_t val_end = obj_str.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          item.name = obj_str.substr(val_start + 1, val_end - val_start - 1);
        }
      }
      
      size_t bg_pos = obj_str.find("\"background\"");
      if (bg_pos != std::string::npos) {
        size_t val_start = obj_str.find('"', bg_pos + 12);
        size_t val_end = obj_str.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          item.background = obj_str.substr(val_start + 1, val_end - val_start - 1);
        }
      }
      
      size_t emb_pos = obj_str.find("\"embedding\"");
      if (emb_pos != std::string::npos) {
        size_t arr_start = obj_str.find('[', emb_pos);
        size_t arr_end = obj_str.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
          std::string arr_str = obj_str.substr(arr_start + 1, arr_end - arr_start - 1);
          std::stringstream ss(arr_str);
          std::string val;
          while (std::getline(ss, val, ',')) {
            if (!val.empty()) {
              item.embedding.push_back(std::stof(val));
            }
          }
        }
      }
      
      if (!item.embedding.empty()) {
        static_gallery_[id] = item;
      }
    }
  }

  std::cout << "[FaceEngine] Loaded " << static_gallery_.size() << " face embedding vectors from gallery.\n";
}

std::vector<FaceDetection> FaceEngine::DetectFaces(const cv::Mat& frame, float det_threshold) {
  std::vector<FaceDetection> detections;
  if (frame.empty()) return detections;

  // Preprocess input size to square det_size
  int det_size = config_.det_size;
  float scale = 1.0F;
  
  float im_ratio = static_cast<float>(frame.rows) / frame.cols;
  float model_ratio = 1.0F;
  int new_w, new_h;
  if (im_ratio > model_ratio) {
    new_h = det_size;
    new_w = static_cast<int>(new_h / im_ratio);
  } else {
    new_w = det_size;
    new_h = static_cast<int>(new_w * im_ratio);
  }
  scale = static_cast<float>(new_h) / frame.rows;

  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(new_w, new_h));
  cv::Mat det_img = cv::Mat::zeros(det_size, det_size, CV_8UC3);
  resized.copyTo(det_img(cv::Rect(0, 0, new_w, new_h)));

  // Convert to float RGB tensor (mean 127.5, std 128.0)
  std::vector<float> input_tensor_values(1 * 3 * det_size * det_size);
  float* input_data = input_tensor_values.data();
  cv::Mat rgb_img;
  cv::cvtColor(det_img, rgb_img, cv::COLOR_BGR2RGB);

  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < det_size; ++h) {
      for (int w = 0; w < det_size; ++w) {
        float val = static_cast<float>(rgb_img.at<cv::Vec3b>(h, w)[c]);
        input_data[c * det_size * det_size + h * det_size + w] = (val - 127.5F) / 128.0F;
      }
    }
  }

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> input_shape = {1, 3, det_size, det_size};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

  const char* input_names[] = {"input.1"};
  auto output_name_strs = impl_->det_session->GetOutputNames();
  std::vector<const char*> output_names;
  for (const auto& name : output_name_strs) {
    output_names.push_back(name.c_str());
  }

  Ort::Value inputs[] = {std::move(input_tensor)};
  auto output_tensors = impl_->det_session->Run(
      Ort::RunOptions{nullptr},
      input_names,
      inputs,
      1,
      output_names.data(),
      output_names.size()
  );

  std::vector<BBox> decoded_boxes;
  std::vector<float> decoded_scores;
  std::vector<std::vector<std::pair<float, float>>> decoded_landmarks;

  int strides[] = {8, 16, 32};
  int fmc = 3;

  for (int idx = 0; idx < 3; ++idx) {
    int stride = strides[idx];
    float* scores_data = output_tensors[idx].GetTensorMutableData<float>();
    float* bbox_data = output_tensors[idx + fmc].GetTensorMutableData<float>();
    float* kps_data = output_tensors[idx + fmc * 2].GetTensorMutableData<float>();

    auto score_shape = output_tensors[idx].GetTensorTypeAndShapeInfo().GetShape();
    int num_anchors = static_cast<int>(score_shape[0]);

    std::vector<std::pair<float, float>> centers = GenerateAnchors(det_size, det_size, stride, 2);

    for (int i = 0; i < num_anchors; ++i) {
      float score = scores_data[i];
      if (score >= det_threshold) {
        float cx = centers[i].first;
        float cy = centers[i].second;

        float dx1 = bbox_data[i * 4 + 0] * stride;
        float dy1 = bbox_data[i * 4 + 1] * stride;
        float dx2 = bbox_data[i * 4 + 2] * stride;
        float dy2 = bbox_data[i * 4 + 3] * stride;

        float x1 = (cx - dx1) / scale;
        float y1 = (cy - dy1) / scale;
        float x2 = (cx + dx2) / scale;
        float y2 = (cy + dy2) / scale;

        std::vector<std::pair<float, float>> lms(5);
        for (int k = 0; k < 5; ++k) {
          float lmx = (cx + kps_data[i * 10 + k * 2 + 0] * stride) / scale;
          float lmy = (cy + kps_data[i * 10 + k * 2 + 1] * stride) / scale;
          lms[k] = {lmx, lmy};
        }

        decoded_boxes.push_back(BBox{x1, y1, x2, y2});
        decoded_scores.push_back(score);
        decoded_landmarks.push_back(lms);
      }
    }
  }

  // Convert to OpenCV format for NMS
  std::vector<cv::Rect> cv_boxes;
  cv_boxes.reserve(decoded_boxes.size());
  for (const auto& b : decoded_boxes) {
    cv_boxes.push_back(cv::Rect(
        static_cast<int>(b.x1),
        static_cast<int>(b.y1),
        static_cast<int>(std::max(1.0F, b.x2 - b.x1)),
        static_cast<int>(std::max(1.0F, b.y2 - b.y1))
    ));
  }

  std::vector<int> indices;
  if (!cv_boxes.empty()) {
    cv::dnn::NMSBoxes(cv_boxes, decoded_scores, det_threshold, 0.4F, indices);
  }

  detections.reserve(indices.size());
  for (int idx : indices) {
    FaceDetection d;
    d.bbox = decoded_boxes[idx];
    d.det_score = decoded_scores[idx];
    d.landmarks = decoded_landmarks[idx];
    
    // Evaluate quality immediately
    float q_score = 0.0F;
    d.quality = ScoreQuality(frame, d.bbox, d.det_score, q_score, d.landmarks);
    d.quality_score = q_score;
    
    detections.push_back(d);
  }

  return detections;
}

std::vector<float> FaceEngine::ExtractEmbedding(const cv::Mat& frame, const std::vector<std::pair<float, float>>& landmarks) {
  if (frame.empty() || landmarks.size() < 5) return std::vector<float>(512, 0.0F);

  // Target coordinates for 112x112 similarity align
  std::vector<cv::Point2f> dst_pts = {
      {30.2946F, 51.6963F}, // Left eye
      {65.5318F, 51.5014F}, // Right eye
      {48.0252F, 71.7366F}, // Nose
      {33.5493F, 92.3655F}, // Left mouth
      {62.7299F, 92.2041F}  // Right mouth
  };

  std::vector<cv::Point2f> src_pts(5);
  for (int i = 0; i < 5; ++i) {
    src_pts[i] = cv::Point2f(landmarks[i].first, landmarks[i].second);
  }

  // Estimate similarity matrix (rotation, translation, scale)
  cv::Mat M = cv::estimateAffinePartial2D(src_pts, dst_pts);
  if (M.empty()) {
    // fallback to first 3 points getAffineTransform
    std::vector<cv::Point2f> src_3 = {src_pts[0], src_pts[1], src_pts[2]};
    std::vector<cv::Point2f> dst_3 = {dst_pts[0], dst_pts[1], dst_pts[2]};
    M = cv::getAffineTransform(src_3, dst_3);
  }

  cv::Mat aligned;
  cv::warpAffine(frame, aligned, M, cv::Size(112, 112));

  // Convert to RGB float tensor (mean 127.5, std 127.5)
  cv::Mat rgb_img;
  cv::cvtColor(aligned, rgb_img, cv::COLOR_BGR2RGB);

  std::vector<float> input_tensor_values(1 * 3 * 112 * 112);
  float* input_data = input_tensor_values.data();

  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < 112; ++h) {
      for (int w = 0; w < 112; ++w) {
        float val = static_cast<float>(rgb_img.at<cv::Vec3b>(h, w)[c]);
        input_data[c * 112 * 112 + h * 112 + w] = (val - 127.5F) / 127.5F;
      }
    }
  }

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> input_shape = {1, 3, 112, 112};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

  const char* input_names[] = {"input.1"};
  const char* output_names[] = {"683"};

  Ort::Value inputs[] = {std::move(input_tensor)};
  auto output_tensors = impl_->rec_session->Run(
      Ort::RunOptions{nullptr},
      input_names,
      inputs,
      1,
      output_names,
      1
  );

  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  std::vector<float> embedding(output_data, output_data + 512);

  return NormalizeEmbedding(embedding);
}

Quality FaceEngine::ScoreQuality(const cv::Mat& frame, const BBox& bbox, float det_score, float& quality_score, const std::vector<std::pair<float, float>>& landmarks) {
  int height = frame.rows;
  int width = frame.cols;

  // Clamp bbox values
  int x1 = std::max(0, std::min(width - 1, static_cast<int>(std::round(bbox.x1))));
  int y1 = std::max(0, std::min(height - 1, static_cast<int>(std::round(bbox.y1))));
  int x2 = std::max(0, std::min(width, static_cast<int>(std::round(bbox.x2))));
  int y2 = std::max(0, std::min(height, static_cast<int>(std::round(bbox.y2))));

  double blur = 0.0;
  if (x2 > x1 && y2 > y1) {
    cv::Mat crop = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    blur = ComputeBlurScore(gray);
  }

  // Ratios
  float bbox_w = std::max(0.0F, bbox.x2 - bbox.x1);
  float bbox_h = std::max(0.0F, bbox.y2 - bbox.y1);
  float area = bbox_w * bbox_h;
  float area_ratio = area / std::max(1, width * height);

  float margin = std::min({bbox.x1, bbox.y1, static_cast<float>(width) - bbox.x2, static_cast<float>(height) - bbox.y2});
  float margin_ratio = std::max(0.0F, margin) / std::max(1, std::min(width, height));

  // eye distance check
  double eye_dist = 0.0;
  if (landmarks.size() >= 2) {
    cv::Point2f pt_left(landmarks[0].first, landmarks[0].second);
    cv::Point2f pt_right(landmarks[1].first, landmarks[1].second);
    eye_dist = cv::norm(pt_left - pt_right);
  }

  // Weight computation components
  float good_det = 0.78F;
  float ok_det = 0.55F;
  float good_blur = 80.0F;
  float ok_blur = 30.0F;
  float min_area_ratio = 0.0025F;
  float edge_margin = 0.01F;

  float det_component = std::min(1.0F, std::max(0.0F, det_score));
  float blur_component = std::min(1.0F, static_cast<float>(blur) / good_blur);
  float size_component = std::min(1.0F, area_ratio / (min_area_ratio * 4.0F));
  float edge_component = std::min(1.0F, margin_ratio / (edge_margin * 2.0F));

  quality_score = (0.42F * det_component +
                   0.24F * blur_component +
                   0.20F * size_component +
                   0.14F * edge_component);

  bool meets_good = (det_score >= good_det &&
                     blur >= good_blur &&
                     area_ratio >= min_area_ratio &&
                     margin_ratio >= edge_margin &&
                     eye_dist >= 12.0);

  if (meets_good) return Quality::kGood;

  bool meets_ok = (det_score >= ok_det &&
                   blur >= ok_blur &&
                   area_ratio >= min_area_ratio * 0.5F &&
                   eye_dist >= 8.0);

  if (meets_ok) return Quality::kOk;

  return Quality::kLow;
}

std::pair<int, float> FaceEngine::AssignIdentity(int track_id, const std::vector<float>& embedding, Quality quality) {
  // 1. If quality is good enough, append to historical feature window to buffer temporal variations
  if (quality == Quality::kGood && !embedding.empty()) {
    auto& history = track_embeddings_history_[track_id];
    history.push_back(embedding);
    if (history.size() > 12) {
      history.erase(history.begin());
    }
  }

  // 2. Compute sliding-window average normalized embedding to overcome single-frame motion blur or profile changes
  std::vector<float> query_embedding = embedding;
  const auto& history = track_embeddings_history_[track_id];
  if (!history.empty()) {
    std::vector<float> sum_emb(512, 0.0F);
    for (const auto& emb : history) {
      for (size_t i = 0; i < 512; ++i) {
        sum_emb[i] += emb[i];
      }
    }
    query_embedding = NormalizeEmbedding(sum_emb);
  }

  // 3. Match against static gallery (highest priority)
  int best_static_id = -1;
  float best_static_sim = -1.0F;

  for (const auto& item : static_gallery_) {
    float sim = CosineSimilarity(query_embedding, item.second.embedding);
    if (sim > best_static_sim) {
      best_static_id = item.first;
      best_static_sim = sim;
    }
  }

  if (best_static_id != -1 && best_static_sim >= config_.cosine_match_threshold) {
    return {best_static_id, best_static_sim};
  }

  // 4. Match against dynamically registered centroids
  int best_dynamic_id = -1;
  float best_dynamic_sim = -1.0F;

  for (const auto& item : dynamic_centroids_) {
    float sim = CosineSimilarity(query_embedding, item.second);
    if (sim > best_dynamic_sim) {
      best_dynamic_id = item.first;
      best_dynamic_sim = sim;
    }
  }

  if (best_dynamic_id != -1 && best_dynamic_sim >= config_.cosine_match_threshold) {
    if (quality == Quality::kGood) {
      // update dynamic centroid vector with running average (90% centroid, 10% new embedding)
      std::vector<float> updated(512);
      for (size_t i = 0; i < 512; ++i) {
        updated[i] = 0.90F * dynamic_centroids_[best_dynamic_id][i] + 0.10F * embedding[i];
      }
      dynamic_centroids_[best_dynamic_id] = NormalizeEmbedding(updated);
      dynamic_counts_[best_dynamic_id]++;
    }
    return {best_dynamic_id, best_dynamic_sim};
  }

  // 5. YOLO26 Joint Tracking Lock & Hysteresis Thresholding
  // If this track has already been successfully bound to an ID, we prioritize locking it
  auto state_it = identity_states_.find(track_id);
  bool already_assigned = (state_it != identity_states_.end() && state_it->second.assigned_id != -1);
  if (already_assigned) {
    int prev_id = state_it->second.assigned_id;
    float prev_sim = -1.0F;
    if (static_gallery_.count(prev_id) > 0) {
      prev_sim = CosineSimilarity(query_embedding, static_gallery_[prev_id].embedding);
    } else if (dynamic_centroids_.count(prev_id) > 0) {
      prev_sim = CosineSimilarity(query_embedding, dynamic_centroids_[prev_id]);
    }

    float hysteresis_threshold = config_.cosine_match_threshold - 0.10F; // typically 0.28
    // Check if there is another extremely strong candidate trying to take over (prevent wrong cross-bindings)
    float competing_sim = std::max(best_static_sim, best_dynamic_sim);
    bool strong_competitor = (competing_sim > 0.65F && (competing_sim - prev_sim) > 0.25F);

    if (prev_sim >= hysteresis_threshold && !strong_competitor) {
      // Direct lock and bypass registration of any new ID to prevent splits
      return {prev_id, prev_sim};
    }
  }

  // 6. Register as new dynamic centroid if face quality is good AND track is not already bound
  if (quality == Quality::kGood && !already_assigned) {
    // Only register if it's genuinely a new face (all similarities to current gallery are low)
    if (best_static_sim < 0.35F && best_dynamic_sim < 0.35F) {
      int new_id = next_dynamic_id_++;
      dynamic_centroids_[new_id] = embedding;
      dynamic_counts_[new_id] = 1;
      std::cout << "[FaceEngine] New dynamic identity registered: I" << new_id 
                << " for track " << track_id << "\n";
      return {new_id, 1.0F};
    }
  }

  // Fallback: If already bound but similarity dropped severely without any competitor, keep the ID but with degraded confidence
  if (already_assigned) {
    int prev_id = state_it->second.assigned_id;
    float prev_sim = 0.0F;
    if (static_gallery_.count(prev_id) > 0) {
      prev_sim = CosineSimilarity(query_embedding, static_gallery_[prev_id].embedding);
    } else if (dynamic_centroids_.count(prev_id) > 0) {
      prev_sim = CosineSimilarity(query_embedding, dynamic_centroids_[prev_id]);
    }
    return {prev_id, prev_sim};
  }

  return {-1, 0.0F};
}

std::vector<std::pair<float, float>> FaceEngine::SmoothLandmarks(int track_id, const std::vector<std::pair<float, float>>& landmarks) {
  // Landmarks smoothing logic (using highly responsive running average filter)
  // Lowers historical weight to 12% to avoid drag/lag under fast head motions
  if (last_observed_faces_.count(track_id) > 0 && last_observed_faces_[track_id].landmarks_5pt.size() == landmarks.size()) {
    std::vector<std::pair<float, float>> smoothed = landmarks;
    const auto& last = last_observed_faces_[track_id].landmarks_5pt;
    for (size_t i = 0; i < landmarks.size(); ++i) {
      smoothed[i].first = 0.12F * last[i].first + 0.88F * landmarks[i].first;
      smoothed[i].second = 0.12F * last[i].second + 0.88F * landmarks[i].second;
    }
    return smoothed;
  }
  return landmarks;
}

std::string FaceEngine::StableQualityLabel(int track_id, float det_score, float blur, float area_ratio, float margin_ratio) {
  if (quality_states_.count(track_id) == 0) {
    QualityState state;
    state.stable_label = "low";
    state.pending_label = "low";
    state.pending_count = 0;
    state.smoothed_blur = blur;
    quality_states_[track_id] = state;
  }

  auto& state = quality_states_[track_id];
  state.smoothed_blur = 0.82 * state.smoothed_blur + 0.18 * blur;

  float good_det = 0.78F;
  float good_blur = 80.0F;
  float min_area_ratio = 0.0025F;
  float edge_margin = 0.01F;
  float ok_det = 0.55F;
  float ok_blur = 30.0F;

  bool is_good = (det_score >= good_det &&
                  state.smoothed_blur >= good_blur &&
                  area_ratio >= min_area_ratio &&
                  margin_ratio >= edge_margin);

  bool is_ok = (det_score >= ok_det &&
                state.smoothed_blur >= ok_blur &&
                area_ratio >= min_area_ratio * 0.5F);

  std::string candidate = is_good ? "good" : (is_ok ? "ok" : "low");

  if (state.stable_label == candidate) {
    state.pending_label = candidate;
    state.pending_count = 0;
    return candidate;
  }

  int required_consecutive = 2;
  if (state.stable_label == "good" && candidate == "ok") {
    // slight drop is tolerated unless persistent
    required_consecutive = 3;
  } else if (state.stable_label == "ok" && candidate == "good") {
    required_consecutive = 3;
  }

  if (state.pending_label == candidate) {
    state.pending_count++;
  } else {
    state.pending_label = candidate;
    state.pending_count = 1;
  }

  if (state.pending_count < required_consecutive) {
    return state.stable_label;
  }

  state.stable_label = candidate;
  state.pending_count = 0;
  return candidate;
}

std::vector<TrackEvent> FaceEngine::UpdateTracks(
    const cv::Mat& frame,
    const std::vector<TrackEvent>& tracks,
    int frame_count,
    std::int64_t capture_timestamp_ms) {
  std::lock_guard<std::mutex> lock(face_mu_);
  std::vector<TrackEvent> out_tracks = tracks;
  if (frame.empty() || out_tracks.empty()) return out_tracks;

  int height = frame.rows;
  int width = frame.cols;

  // 1. Garbage collect stale tracking state maps
  std::unordered_set<int> active_ids;
  for (const auto& t : out_tracks) {
    active_ids.insert(t.person_track_id);
  }

  for (auto it = face_filters_.begin(); it != face_filters_.end();) {
    if (active_ids.count(it->first) == 0) {
      last_update_times_.erase(it->first);
      last_face_relative_boxes_.erase(it->first);
      last_observed_faces_.erase(it->first);
      quality_states_.erase(it->first);
      identity_states_.erase(it->first);
      track_embeddings_history_.erase(it->first);
      it = face_filters_.erase(it);
    } else {
      ++it;
    }
  }

  // 2. Perform secondary face detection on local head ROIs for active person tracks
  // This yields a massive detection accuracy boost compared to full-image downscaled runs.
  bool run_detection = (frame_count % std::max(1, config_.run_every_n_frames) == 0);

  std::map<int, FaceDetection> detections_by_track;

  if (run_detection) {
    struct FaceCandidate {
      int person_track_id = -1;
      FaceDetection detection;
      float match_score = 0.0F;
    };
    std::vector<FaceCandidate> candidates;
    for (auto& track : out_tracks) {
      cv::Rect roi_rect = GetHeadROI(track.bbox, width, height);
      if (roi_rect.width <= 0 || roi_rect.height <= 0) {
        continue;
      }

      // Extract upper body / head region for fine-grained SCRFD face detection
      cv::Mat roi = frame(roi_rect).clone();
      auto roi_faces = DetectFaces(roi, 0.45F);

      int best_idx = -1;
      float best_match_score = -1.0F;
      std::vector<FaceDetection> global_roi_faces;

      for (size_t i = 0; i < roi_faces.size(); ++i) {
        FaceDetection glob_face = roi_faces[i];

        // Project local coordinates back to the global frame coordinate system
        glob_face.bbox.x1 += roi_rect.x;
        glob_face.bbox.y1 += roi_rect.y;
        glob_face.bbox.x2 += roi_rect.x;
        glob_face.bbox.y2 += roi_rect.y;

        for (auto& pt : glob_face.landmarks) {
          pt.first += roi_rect.x;
          pt.second += roi_rect.y;
        }

        // Re-evaluate quality accurately on the global frame resolution to fix ratio margins
        float q_score = 0.0F;
        glob_face.quality = ScoreQuality(frame, glob_face.bbox, glob_face.det_score, q_score, glob_face.landmarks);
        glob_face.quality_score = q_score;
        glob_face.blur_score = static_cast<float>(ComputeFaceBlurScore(frame, glob_face.bbox));

        float match_score = FacePersonMatchScore(glob_face.bbox, track.bbox);
        if (match_score > best_match_score) {
          best_idx = static_cast<int>(global_roi_faces.size());
          best_match_score = match_score;
        }
        global_roi_faces.push_back(glob_face);
      }

      if (best_idx != -1 && best_match_score >= 0.08F) {
        candidates.push_back(FaceCandidate{
            track.person_track_id,
            global_roi_faces[best_idx],
            best_match_score,
        });
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
      return left.match_score > right.match_score;
    });
    std::vector<BBox> assigned_faces;
    for (const auto& candidate : candidates) {
      bool duplicate = false;
      for (const auto& assigned : assigned_faces) {
        if (ComputeIou(candidate.detection.bbox, assigned) >= 0.50F) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        detections_by_track[candidate.person_track_id] = candidate.detection;
        assigned_faces.push_back(candidate.detection.bbox);
      }
    }
  }

  // 3. Process each track's state machine (confirmed / tracking / held / low_quality / unknown)
  const double now_s = capture_timestamp_ms / 1000.0;

  for (auto& track : out_tracks) {
    int pid = track.person_track_id;

    // Track state default initialization
    if (identity_states_.count(pid) == 0) {
      IdentityState s;
      s.assigned_id = -1;
      s.name = "unknown";
      s.state = "unknown";
      s.consecutive_count = 0;
      identity_states_[pid] = s;
    }

    auto& id_state = identity_states_[pid];
    bool face_observed_now = (detections_by_track.count(pid) > 0);

    if (face_observed_now) {
      const auto& det = detections_by_track[pid];

      // Smooth face box using Kalman Filter
      if (face_filters_.count(pid) == 0) {
        face_filters_[pid] = std::make_unique<KalmanFilter>(det.bbox);
        track.face.bbox = det.bbox;
      } else {
        float last_upd = last_update_times_.count(pid) > 0 ? last_update_times_[pid].first : static_cast<float>(now_s - 0.04);
        float dt = std::max(0.001F, std::min(0.5F, static_cast<float>(now_s - last_upd)));
        face_filters_[pid]->Predict(dt);
        track.face.bbox = face_filters_[pid]->Update(det.bbox);
      }
      last_update_times_[pid] = {static_cast<float>(now_s), static_cast<float>(now_s)};

      // Smooth landmarks
      track.face.landmarks_5pt = SmoothLandmarks(pid, det.landmarks);
      track.face.face_bbox_observed = true;
      track.face.face_bbox_last_observed_ms = capture_timestamp_ms;

      // Extract ArcFace embedding if quality permits
      std::vector<float> embedding;
      if (det.quality == Quality::kGood || det.quality == Quality::kOk) {
        embedding = ExtractEmbedding(frame, track.face.landmarks_5pt);
      }

      // Smooth and evaluate quality label
      float body_w = std::max(1.0F, track.bbox.x2 - track.bbox.x1);
      float body_h = std::max(1.0F, track.bbox.y2 - track.bbox.y1);
      float face_w = std::max(1.0F, track.face.bbox.x2 - track.face.bbox.x1);
      float face_h = std::max(1.0F, track.face.bbox.y2 - track.face.bbox.y1);
      
      float area_ratio = (face_w * face_h) / (width * height);
      float margin = std::min({track.face.bbox.x1, track.face.bbox.y1, static_cast<float>(width) - track.face.bbox.x2, static_cast<float>(height) - track.face.bbox.y2});
      float margin_ratio = std::max(0.0F, margin) / std::min(width, height);

      std::string quality_str = StableQualityLabel(pid, det.det_score, det.blur_score, area_ratio, margin_ratio);
      track.face.quality = (quality_str == "good") ? Quality::kGood : ((quality_str == "ok") ? Quality::kOk : Quality::kLow);
      track.face.quality_score = det.quality_score;

      // Run Re-ID recognition matching
      if (!embedding.empty()) {
        auto match = AssignIdentity(pid, embedding, track.face.quality);
        int matched_id = match.first;
        float confidence = match.second;

        if (matched_id != -1) {
          const bool can_switch_identity =
              track.face.quality == Quality::kGood || id_state.assigned_id == matched_id;
          if (can_switch_identity) {
            if (id_state.assigned_id == matched_id) {
              id_state.consecutive_count++;
            } else {
              id_state.assigned_id = matched_id;
              id_state.consecutive_count = 1;
            }

            if (id_state.consecutive_count >= 3) {
              track.face_track_id = matched_id;
              track.identity_confidence = confidence;

              if (static_gallery_.count(matched_id) > 0) {
                id_state.name = static_gallery_[matched_id].name;
                id_state.state = "confirmed";
              } else {
                id_state.name = "I" + std::to_string(matched_id);
                id_state.state = "tracking";
              }
            }
          }
        }
      }

      // Store face-to-body relative ratio
      track.face.face_bbox_observed = true;
      last_face_relative_boxes_[pid] = BBox{
          (track.face.bbox.x1 - track.bbox.x1) / body_w,
          (track.face.bbox.y1 - track.bbox.y1) / body_h,
          (track.face.bbox.x2 - track.bbox.x1) / body_w,
          (track.face.bbox.y2 - track.bbox.y1) / body_h
      };

      // Keep record of last observed face info
      last_observed_faces_[pid] = track.face;
    } else if (!run_detection && last_face_relative_boxes_.count(pid) > 0 && last_observed_faces_.count(pid) > 0) {
      // Skipped detection frame (for performance) - predict face box relative to body movement
      float body_w = std::max(1.0F, track.bbox.x2 - track.bbox.x1);
      float body_h = std::max(1.0F, track.bbox.y2 - track.bbox.y1);
      const auto& rel = last_face_relative_boxes_[pid];
      
      // Keep previous observed face info (quality, landmarks, observation flag)
      track.face = last_observed_faces_[pid];
      
      // Update bbox based on current body track position
      track.face.bbox = BBox{
          track.bbox.x1 + rel.x1 * body_w,
          track.bbox.y1 + rel.y1 * body_h,
          track.bbox.x1 + rel.x2 * body_w,
          track.bbox.y1 + rel.y2 * body_h
      };
      track.face.landmarks_5pt = ProjectLandmarks(
          last_observed_faces_[pid].landmarks_5pt,
          last_observed_faces_[pid].bbox,
          track.face.bbox);
      
      float last_obs_ms = last_observed_faces_[pid].face_bbox_last_observed_ms;
      if (last_obs_ms > 0 && (capture_timestamp_ms - last_obs_ms) <= 600) {
        track.face.face_bbox_observed = true;
      } else {
        track.face.face_bbox_observed = false;
        track.face.quality = Quality::kLow;
      }
      
      // Predict Kalman Filter to advance internal time and velocity states, but do NOT update it
      if (face_filters_.count(pid) > 0) {
        float last_upd = last_update_times_.count(pid) > 0 ? last_update_times_[pid].first : static_cast<float>(now_s - 0.04);
        float dt = std::max(0.001F, std::min(0.5F, static_cast<float>(now_s - last_upd)));
        face_filters_[pid]->Predict(dt);
        // Do NOT call filters_[pid]->Update to bypass double-filtering latency
      }
      last_update_times_[pid] = {static_cast<float>(now_s), static_cast<float>(now_s)};
      
      // Maintain previous identity state
      if (id_state.assigned_id != -1) {
        track.face_track_id = id_state.assigned_id;
      }
      
      // Update relative box again so it stays aligned
      last_face_relative_boxes_[pid] = BBox{
          (track.face.bbox.x1 - track.bbox.x1) / body_w,
          (track.face.bbox.y1 - track.bbox.y1) / body_h,
          (track.face.bbox.x2 - track.bbox.x1) / body_w,
          (track.face.bbox.y2 - track.bbox.y1) / body_h
      };
    } else {
      // OC-SORT / Hysteresis: face is occluded or not detected in this frame.
      // Retain the last known face details but project it relative to the body track movements
      if (last_face_relative_boxes_.count(pid) > 0 && last_observed_faces_.count(pid) > 0) {
        float body_w = std::max(1.0F, track.bbox.x2 - track.bbox.x1);
        float body_h = std::max(1.0F, track.bbox.y2 - track.bbox.y1);
        const auto& rel = last_face_relative_boxes_[pid];
        
        track.face = last_observed_faces_[pid];
        track.face.bbox = BBox{
            track.bbox.x1 + rel.x1 * body_w,
            track.bbox.y1 + rel.y1 * body_h,
            track.bbox.x1 + rel.x2 * body_w,
            track.bbox.y1 + rel.y2 * body_h
        };
        track.face.landmarks_5pt = ProjectLandmarks(
            last_observed_faces_[pid].landmarks_5pt,
            last_observed_faces_[pid].bbox,
            track.face.bbox);
        
        float last_obs_ms = last_observed_faces_[pid].face_bbox_last_observed_ms;
        if (last_obs_ms > 0 && (capture_timestamp_ms - last_obs_ms) <= 600) {
          track.face.face_bbox_observed = true;
          if (id_state.assigned_id != -1) {
            track.face_track_id = id_state.assigned_id;
          }
        } else {
          track.face.face_bbox_observed = false; // not directly observed now
          track.face.quality = Quality::kLow; // degrade quality under occlusion
          if (id_state.assigned_id != -1) {
            track.face_track_id = id_state.assigned_id;
            id_state.state = "held";
          }
        }
      } else {
        // Never observed face, project a default ROI
        track.face.bbox = DefaultProjectFaceBBox(track.bbox);
        track.face.face_bbox_observed = false;
        track.face.quality = Quality::kLow;
        id_state.state = "unknown";
      }
    }

    // 4. Fill final schema fields
    track.body.bbox = track.bbox;
    track.body.quality = track.quality;
    track.body.source = "yolo26_person";

    track.identity_id = (id_state.assigned_id != -1) ? ((static_gallery_.count(id_state.assigned_id) > 0 ? "" : "I") + std::to_string(id_state.assigned_id)) : "";
    track.identity_name = id_state.name;
    track.identity_state = id_state.state;
    
    // Smooth rendering attributes — expand face box for better UI display
    track.render.bbox = ExpandBBox(track.face.bbox, 1.25F, 1.35F, width, height);
    track.render.label = (id_state.assigned_id != -1) ? id_state.name : ("P" + std::to_string(pid));
    
    // Determine if face is close to image boundaries (FOV edge)
    bool is_on_border = (track.face.bbox.x1 < 15.0F ||
                         track.face.bbox.y1 < 15.0F ||
                         track.face.bbox.x2 > width - 15.0F ||
                         track.face.bbox.y2 > height - 15.0F);

    // Determine color state matching: confirmed + good, held + occluded, low_quality, active_speaker
    if (track.face.quality == Quality::kLow || id_state.state == "held" || !track.face.face_bbox_observed || is_on_border) {
      track.render.color_state = "low_quality"; // Gray BBox
      track.render.show_glow = false;
    } else if (track.face.quality == Quality::kGood) {
      track.render.color_state = "confirmed_good"; // Green
      track.render.show_glow = true;
    } else {
      track.render.color_state = "tracking_ok"; // Yellow
      track.render.show_glow = false;
    }

    // Keep body and render geometry separate. API serializers choose render.bbox for UI.
  }

  return out_tracks;
}

} // namespace speaker_id
