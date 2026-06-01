#pragma once

#include "speaker_id/core/types.hpp"

#include <string>
#include <vector>

namespace speaker_id {

struct FusionInput {
  std::vector<TrackEvent> tracks;
  std::vector<VadEvent> vad_segments;
  std::vector<UtteranceEvent> utterances;
  std::vector<ActiveSpeakerScore> active_speaker_scores;
  std::string asd_failure_reason;
};

class FusionBackend {
 public:
  virtual ~FusionBackend() = default;
  virtual std::vector<SpeakerAttribution> Fuse(const FusionInput& input) = 0;
};

class RuleFusionBackend final : public FusionBackend {
 public:
  explicit RuleFusionBackend(
      int frame_width = 1280,
      float active_threshold = 0.55F,
      float visible_threshold = 0.35F,
      float offscreen_threshold = 0.25F,
      float ambiguity_margin = 0.10F,
      bool allow_multi_speaker = true);

  std::vector<SpeakerAttribution> Fuse(const FusionInput& input) override;

 private:
  std::string PositionForTrack(const TrackEvent& track) const;
  float ScoreForPerson(const FusionInput& input, int person_track_id) const;
  bool HasVadEvidence(const FusionInput& input, const UtteranceEvent& utterance) const;

  int frame_width_ = 1280;
  float active_threshold_ = 0.55F;
  float visible_threshold_ = 0.35F;
  float offscreen_threshold_ = 0.25F;
  float ambiguity_margin_ = 0.10F;
  bool allow_multi_speaker_ = true;
};

std::vector<FusionEvent> BuildFusionEvents(
    const FusionInput& input,
    const std::vector<SpeakerAttribution>& attributions);

}  // namespace speaker_id
