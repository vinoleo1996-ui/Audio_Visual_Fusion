#pragma once

#include "speaker_id/core/types.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace speaker_id {

class KalmanFilter {
 public:
  explicit KalmanFilter(const BBox& bbox);
  ~KalmanFilter() = default;

  BBox Predict(float dt);
  BBox Update(const BBox& bbox);

 private:
  std::vector<float> x_; // State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
  std::vector<std::vector<float>> P_; // State covariance matrix
  std::vector<std::vector<float>> Q_; // Process noise covariance
  std::vector<std::vector<float>> R_; // Measurement noise covariance
};

struct Track {
  int person_track_id = -1;
  std::optional<int> face_track_id;
  BBox bbox;
  Quality quality = Quality::kLow;
  std::string track_state = "predicted";
  double last_seen = 0.0;
  int misses = 0;
  std::optional<int> identity_id;
  float confidence = 0.0F;
  int hits = 0;
};

struct PersonDetection {
  BBox bbox;
  float confidence = 0.0F;
};

class SimplePersonTracker {
 public:
  SimplePersonTracker(
      float max_age_s = 1.2F,
      float render_grace_s = 0.35F,
      float high_threshold = 0.35F,
      float low_threshold = 0.15F,
      float iou_threshold = 0.20F,
      int min_hits = 1);

  ~SimplePersonTracker() = default;

  std::vector<TrackEvent> Update(const std::vector<PersonDetection>& detections, int width, int height);
  std::vector<TrackEvent> Update(const std::vector<BBox>& boxes, int width, int height);

 private:
  float max_age_s_;
  float render_grace_s_;
  float high_threshold_;
  float low_threshold_;
  float iou_threshold_;
  int min_hits_;
  int next_id_ = 1;

  std::map<int, Track> tracks_;
  std::map<int, std::unique_ptr<KalmanFilter>> filters_;

  void MergeDuplicateIdentities();
};

float ComputeIou(const BBox& a, const BBox& b);
Quality QualityForBox(const BBox& box, int width, int height, Quality prev_quality = Quality::kLow);

} // namespace speaker_id
