#pragma once

#include "speaker_id/api/vision.hpp"

#include <memory>
#include <string>
#include <vector>

namespace speaker_id {

class YoloVisionBackend : public VisionEngine {
 public:
  YoloVisionBackend(const std::string& model_path, float confidence_threshold = 0.35f);
  ~YoloVisionBackend() override;

  std::vector<TrackEvent> AcceptFrame(const FrameEvent& frame) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace speaker_id
