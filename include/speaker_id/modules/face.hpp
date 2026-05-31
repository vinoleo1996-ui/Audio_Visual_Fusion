#pragma once

#include "speaker_id/core/config.hpp"
#include "speaker_id/core/types.hpp"
#include "speaker_id/modules/kalman_tracker.hpp"

#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <optional>

namespace speaker_id {

struct FaceDetection {
  BBox bbox;
  float det_score = 0.0F;
  std::vector<std::pair<float, float>> landmarks;
  Quality quality = Quality::kLow;
  float quality_score = 0.0F;
  float blur_score = 0.0F;
  std::vector<float> embedding;
};

class FaceEngine {
 public:
  FaceEngine(const FaceConfig& config, const std::string& model_dir, bool use_cuda = false);
  ~FaceEngine();

  // Low-level face detector and extractor interfaces
  std::vector<FaceDetection> DetectFaces(const cv::Mat& frame, float det_threshold = 0.5F);
  std::vector<float> ExtractEmbedding(const cv::Mat& frame, const std::vector<std::pair<float, float>>& landmarks);

  // Quality assessment
  Quality ScoreQuality(const cv::Mat& frame, const BBox& bbox, float det_score, float& quality_score, const std::vector<std::pair<float, float>>& landmarks);

  // Cascading identity matching & tracking stabilizer (returns updated tracks)
  std::vector<TrackEvent> UpdateTracks(
      const cv::Mat& frame,
      const std::vector<TrackEvent>& tracks,
      int frame_count,
      std::int64_t capture_timestamp_ms);

  // Gallery reloading interface
  void ReloadGallery(const std::string& gallery_path = "models/face_gallery.json");

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  FaceConfig config_;
  std::string model_dir_;
  [[maybe_unused]] bool use_cuda_;

  // Thread-safety lock for gallery and centroid updates
  std::mutex face_mu_;

  // Static registered gallery loaded from JSON
  struct GalleryItem {
    std::string name;
    std::string background;
    std::vector<float> embedding;
  };
  std::map<int, GalleryItem> static_gallery_;
  std::string last_gallery_path_;

  // Dynamic discovered identities
  std::map<int, std::vector<float>> dynamic_centroids_;
  std::map<int, int> dynamic_counts_;
  int next_dynamic_id_ = 1;

  // Smoothing filters (Kalman) and relative placement memory
  std::map<int, std::unique_ptr<KalmanFilter>> face_filters_;
  std::map<int, std::pair<float, float>> last_update_times_;
  std::map<int, BBox> last_face_relative_boxes_;
  std::map<int, FaceInfo> last_observed_faces_;
  
  // Track quality state machine (stable_label, pending_label, pending_count)
  struct QualityState {
    std::string stable_label = "low";
    std::string pending_label = "low";
    int pending_count = 0;
    double smoothed_blur = 0.0;
  };
  std::map<int, QualityState> quality_states_;

  // Identity hysteresis voting state (assigned_id, count)
  struct IdentityState {
    int assigned_id = -1; // -1 for unknown
    std::string name = "unknown";
    std::string state = "unknown";
    int consecutive_count = 0;
  };
  std::map<int, IdentityState> identity_states_;

  // Internal helper functions
  std::pair<int, float> AssignIdentity(int track_id, const std::vector<float>& embedding, Quality quality);
  std::vector<std::pair<float, float>> SmoothLandmarks(int track_id, const std::vector<std::pair<float, float>>& landmarks);
  std::string StableQualityLabel(int track_id, float det_score, float blur, float area_ratio, float margin_ratio);

  // Temporal feature buffer per track for smoothing
  std::map<int, std::vector<std::vector<float>>> track_embeddings_history_;
};

} // namespace speaker_id
