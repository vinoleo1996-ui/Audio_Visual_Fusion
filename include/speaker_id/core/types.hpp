#pragma once

#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace speaker_id {

struct BBox {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
};

enum class Quality {
  kGood,
  kOk,
  kLow,
};

struct FrameEvent {
  std::string stream_id;
  std::int64_t frame_id = 0;
  std::int64_t timestamp_ms = 0;
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> data; // contiguous BGR pixels or JPEG bytes
};

struct AudioChunkEvent {
  std::string stream_id;
  std::int64_t timestamp_ms = 0;
  int duration_ms = 0;
  std::int64_t clock_drift_ms = 0;
  double clock_drift_ms_per_min = 0.0;
  bool clock_reanchored = false;
  int sample_rate = 16000;
  std::vector<float> data; // float PCM samples
};

struct FaceInfo {
  std::optional<int> face_track_id;
  BBox bbox;
  std::vector<std::pair<float, float>> landmarks_5pt;
  Quality quality = Quality::kLow;
  float quality_score = 0.0F;
  bool face_bbox_observed = false;
  std::string observation_state = "expired";
  std::int64_t face_bbox_last_observed_ms = 0;
  std::int64_t geometry_timestamp_ms = 0;
  std::int64_t geometry_age_ms = 0;
  bool embedding_eligible = false;
};

struct BodyInfo {
  BBox bbox;
  Quality quality = Quality::kLow;
  std::string source = "yolo26_person";
};

struct RenderInfo {
  BBox bbox;
  std::string label;
  std::string color_state;
  std::string state = "occluded";
  bool show_glow = false;
};

struct TrackEvent {
  int person_track_id = -1;
  std::uint64_t track_snapshot_sequence = 0;
  std::int64_t snapshot_timestamp_ms = 0;
  std::optional<int> face_track_id;
  BBox bbox;
  Quality quality = Quality::kLow;
  float confidence = 0.0F;
  std::string track_state = "predicted";

  // Cascading pipeline fields
  FaceInfo face;
  BodyInfo body;
  RenderInfo render;
  std::string identity_id;
  std::string identity_name;
  std::string identity_state = "unknown";
  float identity_confidence = 0.0F;
};

struct VadEvent {
  std::int64_t start_ms = 0;
  std::int64_t end_ms = 0;
  float confidence = 0.0F;
};

struct UtteranceEvent {
  std::string utterance_id;
  std::int64_t start_ms = 0;
  std::int64_t end_ms = 0;
  std::string text;
  std::string text_delta;
  std::string text_revision = "replace";
  bool final = false;
  float confidence = 0.0F;
  float stability = 0.0F;
  std::vector<std::string> tokens;
  std::vector<float> token_timestamps_s;

  // Diarization fields
  std::string voice_spk_id;
  bool is_overlap = false;
};

struct ActiveSpeakerScore {
  int person_track_id = -1;
  std::optional<int> face_track_id;
  std::int64_t timestamp_ms = 0;
  float p_active = 0.0F;
  float raw_p_active = 0.0F;
  float mean_p_active = 0.0F;
  float peak_p_active = 0.0F;
  float active_ratio = 0.0F;
  int stable_ms = 0;
  int crop_count = 0;
  int max_crop_gap_ms = 0;
  std::int64_t visual_watermark_lag_ms = 0;
  float p_audible_speaking = 0.0F;
  float p_visual_mouth_motion = 0.0F;
  float p_av_sync = 0.0F;
  float face_quality = 0.0F;
  std::vector<std::string> uncertainty_flags;
};

struct SpeakerAttribution {
  std::string utterance_id;
  std::vector<int> person_track_ids;
  std::string position;
  float confidence = 0.0F;
  bool tentative = true;
  std::string decision_reason;
};

struct FusionTrackView {
  int person_track_id = -1;
  std::uint64_t track_snapshot_sequence = 0;
  std::int64_t snapshot_timestamp_ms = 0;
  std::optional<int> face_track_id;
  BBox bbox;
  Quality quality = Quality::kLow;
  float p_active = 0.0F;

  // Cascading pipeline fields
  FaceInfo face;
  BodyInfo body;
  RenderInfo render;
  std::string identity_id;
  std::string identity_name;
  std::string identity_state = "unknown";
  float identity_confidence = 0.0F;
};

struct FusionEvent {
  std::uint64_t event_sequence = 0;
  std::string event_type;
  std::string utterance_id;
  std::int64_t start_ms = 0;
  std::int64_t end_ms = 0;
  std::string text;
  std::string text_delta;
  std::string text_revision = "replace";
  bool final = false;
  std::string position;
  std::vector<int> person_track_ids;
  float confidence = 0.0F;
  float stability = 0.0F;
  std::vector<float> token_timestamps_s;
  bool tentative = true;
  std::string decision_reason;
  std::vector<FusionTrackView> tracks;

  // Diarization resolved speaker fields
  std::string voice_spk_id;
  std::string speaker_id;
  std::string speaker_name;
};

struct PipelineMetrics {
  std::size_t frame_queue_depth = 0;
  std::size_t frame_queue_dropped = 0;
  std::size_t face_queue_depth = 0;
  std::size_t face_queue_dropped = 0;
  std::size_t person_detector_queue_depth = 0;
  std::size_t person_detector_queue_dropped = 0;
  std::size_t audio_queue_depth = 0;
  std::uint64_t video_frames_processed = 0;
  std::uint64_t person_detector_runs = 0;
  std::uint64_t face_pipeline_runs = 0;
  std::uint64_t asd_windows_scored = 0;
  double video_fps = 0.0;
  double person_detector_fps = 0.0;
  double face_pipeline_fps = 0.0;
  double asd_window_fps = 0.0;
  double person_detector_latency_p50_ms = 0.0;
  double person_detector_latency_p95_ms = 0.0;
  double face_pipeline_latency_p50_ms = 0.0;
  double face_pipeline_latency_p95_ms = 0.0;
  double asd_latency_p50_ms = 0.0;
  double asd_latency_p95_ms = 0.0;
  std::int64_t latest_audio_watermark_ms = 0;
  std::int64_t latest_visual_watermark_ms = 0;
  std::int64_t visual_watermark_lag_ms = 0;
  float direct_face_observation_ratio = 0.0F;
  float asd_usable_window_ratio = 0.0F;
  bool sync_healthy = true;
  double watermark_lag_ms = 0.0;
  double drift_ms_per_min = 0.0;
  std::uint64_t sync_reanchor_count = 0;
  std::uint64_t asd_gated_windows = 0;
  std::uint64_t asd_vad_skipped_windows = 0;
  std::size_t person_track_count = 0;
  std::size_t rendered_face_track_count = 0;
  std::size_t asd_candidate_count = 0;
  std::size_t no_face_track_count = 0;
  std::uint64_t stale_geometry_rejected = 0;
  std::uint64_t preview_frames_encoded = 0;
  std::uint64_t preview_frames_dropped = 0;
  std::uint64_t preview_ws_dropped_frames = 0;
  std::uint64_t pipe_frames_dropped = 0;
  std::size_t pipe_backlog_bytes = 0;
  double preview_encode_fps = 0.0;
  double preview_encode_latency_p95_ms = 0.0;
  double preview_capture_to_send_p95_ms = 0.0;
  std::string preview_frame_age_source = "arrival_estimate";
  bool session_replay_enabled = false;
};

}  // namespace speaker_id
