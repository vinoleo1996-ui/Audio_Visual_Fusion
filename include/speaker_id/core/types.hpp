#pragma once

#include <cstdint>
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
  std::vector<std::uint8_t> data; // raw image data (e.g. RGB or JPEG)
};

struct AudioChunkEvent {
  std::string stream_id;
  std::int64_t timestamp_ms = 0;
  int duration_ms = 0;
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
  std::int64_t face_bbox_last_observed_ms = 0;
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
  bool show_glow = false;
};

struct TrackEvent {
  int person_track_id = -1;
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
};

struct FusionTrackView {
  int person_track_id = -1;
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
  std::vector<FusionTrackView> tracks;

  // Diarization resolved speaker fields
  std::string speaker_id;
  std::string speaker_name;
};

}  // namespace speaker_id
