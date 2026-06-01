#include "speaker_id/modules/kalman_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>

namespace speaker_id {
namespace {

double MonotonicNowSeconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(now).count();
}

float NormalizedCenterDistance(const BBox& left, const BBox& right) {
  const float left_width = std::max(1.0F, left.x2 - left.x1);
  const float left_height = std::max(1.0F, left.y2 - left.y1);
  const float right_width = std::max(1.0F, right.x2 - right.x1);
  const float right_height = std::max(1.0F, right.y2 - right.y1);
  const float dx = (left.x1 + left.x2 - right.x1 - right.x2) * 0.5F;
  const float dy = (left.y1 + left.y2 - right.y1 - right.y2) * 0.5F;
  const float scale =
      std::max(1.0F, std::sqrt((left_width * left_width + left_height * left_height +
                                right_width * right_width + right_height * right_height) *
                               0.5F));
  return std::sqrt(dx * dx + dy * dy) / scale;
}

bool HasCompatibleScale(const BBox& left, const BBox& right) {
  const float left_area =
      std::max(1.0F, (left.x2 - left.x1) * (left.y2 - left.y1));
  const float right_area =
      std::max(1.0F, (right.x2 - right.x1) * (right.y2 - right.y1));
  const float ratio = left_area / right_area;
  return ratio >= 0.20F && ratio <= 5.0F;
}

}  // namespace

float ComputeIou(const BBox& a, const BBox& b) {
  float x1 = std::max(a.x1, b.x1);
  float y1 = std::max(a.y1, b.y1);
  float x2 = std::min(a.x2, b.x2);
  float y2 = std::min(a.y2, b.y2);
  float inter = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  float union_area = area_a + area_b - inter;
  return union_area > 0.0F ? inter / union_area : 0.0F;
}

Quality QualityForBox(const BBox& box, int width, int height, Quality prev_quality) {
  float area = std::max(0.0F, box.x2 - box.x1) * std::max(0.0F, box.y2 - box.y1);
  float area_ratio = area / static_cast<float>(std::max(1, width * height));
  float margin = std::min({box.x1, box.y1, static_cast<float>(width) - box.x2, static_cast<float>(height) - box.y2});
  float margin_thresh = static_cast<float>(std::min(width, height)) * 0.02F;

  float good_area_thresh = (prev_quality == Quality::kGood) ? 0.085F : 0.10F;
  float good_margin_thresh = (prev_quality == Quality::kGood) ? margin_thresh * 0.82F : margin_thresh;

  if (area_ratio >= good_area_thresh && margin >= good_margin_thresh) {
    return Quality::kGood;
  }

  float ok_area_thresh = (prev_quality == Quality::kOk) ? 0.03F : 0.035F;
  if (area_ratio >= ok_area_thresh) {
    return Quality::kOk;
  }

  return Quality::kLow;
}

// Kalman Filter Implementation using pure C++ vectors
KalmanFilter::KalmanFilter(const BBox& bbox) {
  x_ = {bbox.x1, bbox.y1, bbox.x2, bbox.y2, 0.0F, 0.0F, 0.0F, 0.0F};
  
  P_ = std::vector<std::vector<float>>(8, std::vector<float>(8, 0.0F));
  for (int i = 0; i < 8; ++i) {
    P_[i][i] = (i < 4) ? 10.0F : 1000.0F;
  }

  Q_ = std::vector<std::vector<float>>(8, std::vector<float>(8, 0.0F));
  for (int i = 0; i < 8; ++i) {
    Q_[i][i] = (i < 4) ? 0.05F : 0.01F;
  }

  R_ = std::vector<std::vector<float>>(4, std::vector<float>(4, 0.0F));
  for (int i = 0; i < 4; ++i) {
    R_[i][i] = 1.5F;
  }
}

BBox KalmanFilter::Predict(float dt) {
  dt = std::max(0.001F, std::min(0.5F, dt));
  
  // Transition step: self.x = F @ self.x
  x_[0] += dt * x_[4];
  x_[1] += dt * x_[5];
  x_[2] += dt * x_[6];
  x_[3] += dt * x_[7];

  // Predict covariance P = F @ P @ F.T + Q
  std::vector<std::vector<float>> FP(8, std::vector<float>(8, 0.0F));
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      FP[i][j] = (i < 4) ? (P_[i][j] + dt * P_[i + 4][j]) : P_[i][j];
    }
  }

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      P_[i][j] = (j < 4) ? (FP[i][j] + dt * FP[i][j + 4]) : FP[i][j];
      P_[i][j] += Q_[i][j];
    }
  }

  float w = std::max(1.0F, x_[2] - x_[0]);
  float h = std::max(1.0F, x_[3] - x_[1]);
  return BBox{x_[0], x_[1], x_[0] + w, x_[1] + h};
}

BBox KalmanFilter::Update(const BBox& bbox) {
  // Measurement matrix H is just the identity matrix for the first 4 elements.
  // Innovation covariance S = H * P * H_T + R
  std::vector<std::vector<float>> S(4, std::vector<float>(4, 0.0F));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      S[i][j] = P_[i][j] + R_[i][j];
    }
  }

  // Invert 4x4 matrix S (using simple Gaussian elimination)
  std::vector<std::vector<float>> S_inv(4, std::vector<float>(4, 0.0F));
  for (int i = 0; i < 4; ++i) S_inv[i][i] = 1.0F;
  auto S_temp = S;
  for (int i = 0; i < 4; ++i) {
    float pivot = S_temp[i][i];
    if (std::fabs(pivot) < 1e-6F) pivot = 1e-6F;
    for (int j = 0; j < 4; ++j) {
      S_temp[i][j] /= pivot;
      S_inv[i][j] /= pivot;
    }
    for (int k = 0; k < 4; ++k) {
      if (k == i) continue;
      float factor = S_temp[k][i];
      for (int j = 0; j < 4; ++j) {
        S_temp[k][j] -= factor * S_temp[i][j];
        S_inv[k][j] -= factor * S_inv[i][j];
      }
    }
  }

  // Kalman Gain K = P * H_T * S_inv
  std::vector<std::vector<float>> K(8, std::vector<float>(4, 0.0F));
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        K[i][j] += P_[i][k] * S_inv[k][j];
      }
    }
  }

  // State update: x = x + K @ (z - H @ x)
  std::vector<float> z = {bbox.x1, bbox.y1, bbox.x2, bbox.y2};
  std::vector<float> innovation(4);
  for (int i = 0; i < 4; ++i) {
    innovation[i] = z[i] - x_[i];
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      x_[i] += K[i][j] * innovation[j];
    }
  }

  // Covariance update: P = (I - K @ H) @ P
  std::vector<std::vector<float>> KH_P(8, std::vector<float>(8, 0.0F));
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int k = 0; k < 4; ++k) {
        KH_P[i][j] += K[i][k] * P_[k][j];
      }
    }
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      P_[i][j] -= KH_P[i][j];
    }
  }

  if (x_[2] < x_[0]) x_[2] = x_[0] + 1.0F;
  if (x_[3] < x_[1]) x_[3] = x_[1] + 1.0F;

  return BBox{x_[0], x_[1], x_[2], x_[3]};
}

SimplePersonTracker::SimplePersonTracker(
    float max_age_s,
    float render_grace_s,
    float high_threshold,
    float low_threshold,
    float iou_threshold,
    int min_hits,
    float center_distance_threshold)
    : max_age_s_(max_age_s),
      render_grace_s_(render_grace_s),
      high_threshold_(high_threshold),
      low_threshold_(low_threshold),
      iou_threshold_(iou_threshold),
      center_distance_threshold_(center_distance_threshold),
      min_hits_(std::max(1, min_hits)) {}

void SimplePersonTracker::MergeDuplicateIdentities() {
  std::vector<int> to_remove;
  for (auto& [id_a, track_a] : tracks_) {
    if (std::find(to_remove.begin(), to_remove.end(), id_a) != to_remove.end()) continue;
    if (!track_a.identity_id.has_value()) continue;

    for (auto& [id_b, track_b] : tracks_) {
      if (id_a == id_b) continue;
      if (std::find(to_remove.begin(), to_remove.end(), id_b) != to_remove.end()) continue;
      if (!track_b.identity_id.has_value()) continue;

      if (track_a.identity_id == track_b.identity_id) {
        int keep_id = std::min(id_a, id_b);
        int drop_id = std::max(id_a, id_b);

        auto& keep = tracks_[keep_id];
        auto& drop = tracks_[drop_id];

        keep.bbox = drop.bbox;
        keep.quality = drop.quality;
        keep.last_seen = std::max(keep.last_seen, drop.last_seen);
        keep.last_predict = std::max(keep.last_predict, drop.last_predict);
        keep.misses = std::min(keep.misses, drop.misses);
        keep.hits = std::max(keep.hits, drop.hits);
        keep.track_state = drop.track_state;

        if (filters_.count(drop_id) > 0) {
          filters_[keep_id] = std::move(filters_[drop_id]);
        }
        to_remove.push_back(drop_id);
      }
    }
  }

  for (int id : to_remove) {
    tracks_.erase(id);
    filters_.erase(id);
  }
}

std::vector<TrackEvent> SimplePersonTracker::Update(const std::vector<BBox>& boxes, int width, int height) {
  std::vector<PersonDetection> detections;
  detections.reserve(boxes.size());
  for (const auto& box : boxes) {
    detections.push_back(PersonDetection{box, 1.0F});
  }
  return Update(detections, width, height);
}

std::vector<TrackEvent> SimplePersonTracker::Update(
    const std::vector<PersonDetection>& detections, int width, int height) {
  MergeDuplicateIdentities();
  const double now = MonotonicNowSeconds();

  // Step 1: Partition detections into high and low confidence groups
  std::vector<PersonDetection> high_dets;
  std::vector<PersonDetection> low_dets;
  for (const auto& detection : detections) {
    if (detection.confidence >= high_threshold_) {
      high_dets.push_back(detection);
    } else if (detection.confidence >= low_threshold_) {
      low_dets.push_back(detection);
    }
  }

  // Step 2: Predict current expected position for each track via Kalman Filter
  std::map<int, BBox> predictions;
  for (auto& [track_id, track] : tracks_) {
    const double previous_predict = track.last_predict > 0.0 ? track.last_predict : track.last_seen;
    float dt = static_cast<float>(now - previous_predict);
    if (filters_.count(track_id) > 0) {
      predictions[track_id] = filters_[track_id]->Predict(dt);
    } else {
      predictions[track_id] = track.bbox;
    }
    track.last_predict = now;
  }

  std::vector<int> candidate_ids;
  for (const auto& [track_id, _] : tracks_) {
    candidate_ids.push_back(track_id);
  }

  // Step 3: First stage association: match tracks with high confidence detections
  std::vector<int> matched_tracks;
  std::vector<int> matched_high_dets;

  if (!candidate_ids.empty() && !high_dets.empty()) {
    // Greedy linear assignment matcher used when scipy-style LAP is unavailable.
    struct MatchCandidate {
      int candidate_idx;
      int det_idx;
      float cost;
    };
    std::vector<MatchCandidate> options;
    for (size_t i = 0; i < candidate_ids.size(); ++i) {
      int track_id = candidate_ids[i];
      BBox pred_box = predictions[track_id];
      for (size_t j = 0; j < high_dets.size(); ++j) {
        const float iou = ComputeIou(pred_box, high_dets[j].bbox);
        const float center_distance =
            NormalizedCenterDistance(pred_box, high_dets[j].bbox);
        const bool motion_recovery =
            center_distance <= center_distance_threshold_ &&
            HasCompatibleScale(pred_box, high_dets[j].bbox);
        float cost = iou >= iou_threshold_ ? 1.0F - iou : 1.0F + center_distance;
        if (iou >= iou_threshold_ || motion_recovery) {
          options.push_back({static_cast<int>(i), static_cast<int>(j), cost});
        }
      }
    }

    std::sort(options.begin(), options.end(), [](const MatchCandidate& a, const MatchCandidate& b) {
      return a.cost < b.cost;
    });

    std::vector<bool> candidate_used(candidate_ids.size(), false);
    std::vector<bool> det_used(high_dets.size(), false);

    for (const auto& opt : options) {
      if (!candidate_used[opt.candidate_idx] && !det_used[opt.det_idx]) {
        candidate_used[opt.candidate_idx] = true;
        det_used[opt.det_idx] = true;

        int track_id = candidate_ids[opt.candidate_idx];
        const auto& detection = high_dets[opt.det_idx];
        const BBox& det = detection.bbox;

        BBox smoothed_box = det;
        if (filters_.count(track_id) == 0) {
          filters_[track_id] = std::make_unique<KalmanFilter>(det);
        } else {
          smoothed_box = filters_[track_id]->Update(det);
        }

        auto& track = tracks_[track_id];
        track.bbox = smoothed_box;
        track.quality = QualityForBox(smoothed_box, width, height, track.quality);
        track.last_seen = now;
        track.last_predict = now;
        track.misses = 0;
        track.hits++;
        track.track_state = "tracked";
        track.confidence = detection.confidence;

        matched_tracks.push_back(track_id);
        matched_high_dets.push_back(opt.det_idx);
      }
    }
  }

  // Step 4: Second stage association: match remaining tracks with low confidence detections
  std::vector<int> remaining_candidates;
  for (int track_id : candidate_ids) {
    if (std::find(matched_tracks.begin(), matched_tracks.end(), track_id) == matched_tracks.end()) {
      remaining_candidates.push_back(track_id);
    }
  }

  if (!remaining_candidates.empty() && !low_dets.empty()) {
    struct MatchCandidate {
      int candidate_idx;
      int det_idx;
      float cost;
    };
    std::vector<MatchCandidate> options;
    for (size_t i = 0; i < remaining_candidates.size(); ++i) {
      int track_id = remaining_candidates[i];
      BBox pred_box = predictions[track_id];
      for (size_t j = 0; j < low_dets.size(); ++j) {
        const float iou = ComputeIou(pred_box, low_dets[j].bbox);
        const float center_distance =
            NormalizedCenterDistance(pred_box, low_dets[j].bbox);
        const bool motion_recovery =
            center_distance <= center_distance_threshold_ * 0.75F &&
            HasCompatibleScale(pred_box, low_dets[j].bbox);
        float cost = iou >= iou_threshold_ ? 1.0F - iou : 1.0F + center_distance;
        if (iou >= iou_threshold_ || motion_recovery) {
          options.push_back({static_cast<int>(i), static_cast<int>(j), cost});
        }
      }
    }

    std::sort(options.begin(), options.end(), [](const MatchCandidate& a, const MatchCandidate& b) {
      return a.cost < b.cost;
    });

    std::vector<bool> candidate_used(remaining_candidates.size(), false);
    std::vector<bool> det_used(low_dets.size(), false);

    for (const auto& opt : options) {
      if (!candidate_used[opt.candidate_idx] && !det_used[opt.det_idx]) {
        candidate_used[opt.candidate_idx] = true;
        det_used[opt.det_idx] = true;

        int track_id = remaining_candidates[opt.candidate_idx];
        const auto& detection = low_dets[opt.det_idx];
        const BBox& det = detection.bbox;

        BBox smoothed_box = det;
        if (filters_.count(track_id) == 0) {
          filters_[track_id] = std::make_unique<KalmanFilter>(det);
        } else {
          smoothed_box = filters_[track_id]->Update(det);
        }

        auto& track = tracks_[track_id];
        track.bbox = smoothed_box;
        track.quality = QualityForBox(smoothed_box, width, height, track.quality);
        track.last_seen = now;
        track.last_predict = now;
        track.misses = 0;
        track.hits++;
        track.track_state = "tracked";
        track.confidence = detection.confidence;

        matched_tracks.push_back(track_id);
      }
    }
  }

  // Step 5: Mark remaining unmatched tracks as predicted/lost and increment misses
  for (int track_id : candidate_ids) {
    if (std::find(matched_tracks.begin(), matched_tracks.end(), track_id) != matched_tracks.end()) {
      continue;
    }
    auto& track = tracks_[track_id];
    track.misses++;
    track.track_state = "predicted";
    track.bbox = predictions[track_id];
    track.quality = Quality::kLow;
  }

  // Step 6: Initialize new tracks for unmatched high confidence detections
  for (size_t j = 0; j < high_dets.size(); ++j) {
    if (std::find(matched_high_dets.begin(), matched_high_dets.end(), static_cast<int>(j)) != matched_high_dets.end()) {
      continue;
    }
    Track track;
    track.person_track_id = next_id_;
    track.bbox = high_dets[j].bbox;
    track.quality = QualityForBox(high_dets[j].bbox, width, height);
    track.track_state = "tracked";
    track.last_seen = now;
    track.last_predict = now;
    track.misses = 0;
    track.hits = 1;
    track.confidence = high_dets[j].confidence;

    tracks_[next_id_] = track;
    filters_[next_id_] = std::make_unique<KalmanFilter>(high_dets[j].bbox);
    next_id_++;
  }

  // Step 7: Delete stale tracks
  std::vector<int> stale;
  for (const auto& [track_id, track] : tracks_) {
    if (now - track.last_seen > max_age_s_) {
      stale.push_back(track_id);
    }
  }
  for (int track_id : stale) {
    tracks_.erase(track_id);
    filters_.erase(track_id);
  }

  return VisibleTracks(now);
}

std::vector<TrackEvent> SimplePersonTracker::PredictOnly(int width, int height) {
  (void)width;
  (void)height;
  MergeDuplicateIdentities();
  const double now = MonotonicNowSeconds();
  for (auto& [track_id, track] : tracks_) {
    const double previous_predict = track.last_predict > 0.0 ? track.last_predict : track.last_seen;
    const float dt = static_cast<float>(now - previous_predict);
    if (filters_.count(track_id) > 0) {
      track.bbox = filters_[track_id]->Predict(dt);
    }
    track.last_predict = now;
  }
  return VisibleTracks(now);
}

void SimplePersonTracker::ApplyIdentityHints(const std::vector<TrackEvent>& tracks) {
  for (const auto& event : tracks) {
    if (!event.face_track_id.has_value() || event.identity_state == "unknown") {
      continue;
    }
    const auto track = tracks_.find(event.person_track_id);
    if (track == tracks_.end()) {
      continue;
    }
    track->second.face_track_id = event.face_track_id;
    track->second.identity_id = event.face_track_id;
  }
}

std::vector<TrackEvent> SimplePersonTracker::VisibleTracks(double now) const {
  std::vector<TrackEvent> visible;
  for (const auto& [track_id, track] : tracks_) {
    if (track.hits >= min_hits_ &&
        (track.track_state == "tracked" || now - track.last_seen <= render_grace_s_)) {
      TrackEvent ev;
      ev.person_track_id = track.person_track_id;
      ev.face_track_id = track.face_track_id;
      ev.bbox = track.bbox;
      ev.quality = track.quality;
      ev.confidence = track.confidence;
      ev.track_state = track.track_state;
      visible.push_back(ev);
    }
  }

  // Sort by bbox.x1
  std::sort(visible.begin(), visible.end(), [](const TrackEvent& a, const TrackEvent& b) {
    return a.bbox.x1 < b.bbox.x1;
  });

  return visible;
}

} // namespace speaker_id
