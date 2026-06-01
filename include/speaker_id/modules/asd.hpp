#pragma once

#include "speaker_id/core/types.hpp"

#include <map>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace speaker_id {

std::vector<float> ExtractLrAsdMfcc(const std::vector<float>& samples, int sample_rate);

struct AsdTimelinePoint {
  std::int64_t timestamp_ms = 0;
  std::vector<ActiveSpeakerScore> scores;
};

std::vector<ActiveSpeakerScore> AggregateAsdScoresForUtterance(
    const UtteranceEvent& utterance,
    const std::deque<AsdTimelinePoint>& timeline,
    const std::vector<ActiveSpeakerScore>& latest_scores,
    float active_threshold,
    int hop_ms);

struct AsdVisualSample {
  std::int64_t timestamp_ms = 0;
  std::vector<float> grayscale_112;
};

struct AsdInputWindow {
  std::int64_t start_ms = 0;
  std::int64_t end_ms = 0;
  int sample_rate = 16000;
  bool speech_active = false;
  std::vector<float> audio_samples;
  std::map<int, std::vector<AsdVisualSample>> visual_samples;
  std::vector<TrackEvent> tracks;
  std::vector<VadEvent> vad_segments;
};

class AsdBackend {
 public:
  virtual ~AsdBackend() = default;
  virtual std::vector<ActiveSpeakerScore> ScoreWindow(const AsdInputWindow& window) = 0;
};

class SimpleAsdBackend final : public AsdBackend {
 public:
  SimpleAsdBackend() = default;
  ~SimpleAsdBackend() override = default;

  std::vector<ActiveSpeakerScore> ScoreWindow(const AsdInputWindow& window) override;
};

class OrtLrAsdBackend final : public AsdBackend {
 public:
  explicit OrtLrAsdBackend(const std::string& model_path);
  ~OrtLrAsdBackend() override;

  std::vector<ActiveSpeakerScore> ScoreWindow(const AsdInputWindow& window) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace speaker_id
