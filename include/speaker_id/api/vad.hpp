#pragma once

#include "speaker_id/core/types.hpp"
#include <vector>

namespace speaker_id {

class VadEngine {
 public:
  virtual ~VadEngine() = default;
  virtual std::vector<VadEvent> AcceptAudio(const AudioChunkEvent& chunk) = 0;
  virtual bool IsSpeechActive() const = 0;
};

}  // namespace speaker_id
