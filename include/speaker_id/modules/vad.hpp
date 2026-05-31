#pragma once

#include "speaker_id/api/vad.hpp"

#include <memory>
#include <string>
#include <vector>

namespace speaker_id {

class SileroVadBackend : public VadEngine {
 public:
  SileroVadBackend(const std::string& model_path,
                   float threshold = 0.25f,
                   int preroll_ms = 480,
                   int min_silence_ms = 1200,
                   int min_speech_ms = 450,
                   int max_utterance_ms = 12000);
  ~SileroVadBackend() override;

  std::vector<VadEvent> AcceptAudio(const AudioChunkEvent& chunk) override;
  bool IsSpeechActive() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace speaker_id
