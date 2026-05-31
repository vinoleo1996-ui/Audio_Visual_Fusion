#pragma once

#include "speaker_id/core/types.hpp"
#include <string>
#include <vector>

namespace speaker_id {

struct DiarizationSegment {
  int left_sample = 0;
  int right_sample = 0;
  std::string voice_spk_id;
  bool is_overlap = false;
  std::vector<std::string> overlap_speakers;
  float confidence = 0.0F;
};

class DiarizationEngine {
 public:
  virtual ~DiarizationEngine() = default;

  // Run overlap-aware segmentation on an audio utterance and return diarization segments
  virtual std::vector<DiarizationSegment> ProcessUtterance(const std::vector<float>& samples, int sample_rate) = 0;

  // Map a voice speaker ID to a visual identity using the probability affinity matrix
  virtual std::string ResolveSpeaker(const std::string& voice_spk_id, const std::vector<TrackEvent>& visible_tracks) = 0;

  // Update voice-face affinity with a confirmed match from fusion
  virtual void UpdateAffinity(const std::string& voice_spk_id, int person_track_id, float evidence, const std::vector<TrackEvent>& visible_tracks) = 0;
};

}  // namespace speaker_id
