#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace speaker_id {

struct CapturedVideoFrame {
  cv::Mat bgr;
  std::int64_t capture_timestamp_ms = 0;
  std::uint64_t sequence = 0;
};

class LibavVideoCapture {
 public:
  LibavVideoCapture();
  ~LibavVideoCapture();

  bool Open(const std::string& device_name, int width, int height, int fps,
            std::string& error);
  bool ReadFrame(CapturedVideoFrame& frame, std::string& error);
  void Close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace speaker_id
