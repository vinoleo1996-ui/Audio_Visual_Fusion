#pragma once

#include "speaker_id/api/diarization.hpp"

#include <memory>
#include <string>
#include <vector>

namespace speaker_id {

class OrtDiarizationBackend : public DiarizationEngine {
 public:
  OrtDiarizationBackend(const std::string& pyannote_model_path,
                        const std::string& cam_model_path,
                        float segmentation_threshold = 0.50f,
                        int min_segment_ms = 200);
  ~OrtDiarizationBackend() override;

  std::vector<DiarizationSegment> ProcessUtterance(const std::vector<float>& samples, int sample_rate) override;

  std::string ResolveSpeaker(const std::string& voice_spk_id, const std::vector<TrackEvent>& visible_tracks) override;

  void UpdateAffinity(const std::string& voice_spk_id, int person_track_id, float evidence, const std::vector<TrackEvent>& visible_tracks) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace speaker_id
