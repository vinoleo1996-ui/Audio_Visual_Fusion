#pragma once

#include "speaker_id/core/types.hpp"
#include <string>
#include <vector>

namespace speaker_id {

struct AsrResult {
  std::string text;
  float confidence = 0.0F;
  float stability = 0.0F;
  std::string language;
  bool is_hallucination = false;
  std::vector<std::string> tokens;
  std::vector<float> token_timestamps_s;
};

class AsrEngine {
 public:
  virtual ~AsrEngine() = default;
  
  // Offline/turn-level transcription (e.g. SenseVoice)
  virtual AsrResult Transcribe(const std::vector<float>& samples, int sample_rate) = 0;

  // Streaming/real-time partial transcription (e.g. Zipformer)
  virtual void BeginStreaming() = 0;
  virtual AsrResult StreamAudio(const std::vector<float>& samples, int sample_rate) = 0;
  virtual AsrResult EndStreaming() = 0;
};

}  // namespace speaker_id
