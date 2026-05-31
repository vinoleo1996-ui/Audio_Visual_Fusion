#pragma once

#include "speaker_id/core/types.hpp"
#include <vector>

namespace speaker_id {

class VisionEngine {
 public:
  virtual ~VisionEngine() = default;
  virtual std::vector<TrackEvent> AcceptFrame(const FrameEvent& frame) = 0;
};

}  // namespace speaker_id
