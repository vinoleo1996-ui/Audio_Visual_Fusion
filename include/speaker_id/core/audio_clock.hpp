#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace speaker_id {

struct AudioClockStamp {
  std::int64_t timestamp_ms = 0;
  std::int64_t drift_ms = 0;
  double drift_ms_per_min = 0.0;
  bool reanchored = false;
};

class AudioMonotonicPll {
 public:
  AudioMonotonicPll(int reanchor_threshold_ms, double alpha,
                    int max_correction_per_chunk_ms)
      : reanchor_threshold_ms_(std::max(1, reanchor_threshold_ms)),
        alpha_(std::clamp(alpha, 0.0, 1.0)),
        max_correction_per_chunk_ms_(
            std::max(0, max_correction_per_chunk_ms)) {}

  AudioClockStamp Stamp(std::int64_t arrival_end_ms, int duration_ms) {
    const auto safe_duration_ms = std::max(1, duration_ms);
    const auto observed_start_ms = arrival_end_ms - safe_duration_ms;
    if (!initialized_) {
      initialized_ = true;
      next_timestamp_ms_ = observed_start_ms;
      first_arrival_end_ms_ = arrival_end_ms;
    }

    AudioClockStamp stamp;
    stamp.drift_ms = observed_start_ms - next_timestamp_ms_;
    if (std::llabs(stamp.drift_ms) >= reanchor_threshold_ms_) {
      stamp.timestamp_ms = observed_start_ms;
      stamp.reanchored = true;
    } else {
      const auto correction_ms = static_cast<std::int64_t>(std::llround(
          std::clamp(stamp.drift_ms * alpha_,
                     -static_cast<double>(max_correction_per_chunk_ms_),
                     static_cast<double>(max_correction_per_chunk_ms_))));
      stamp.timestamp_ms = next_timestamp_ms_ + correction_ms;
    }
    next_timestamp_ms_ = stamp.timestamp_ms + safe_duration_ms;

    const auto elapsed_ms = std::max<std::int64_t>(
        1, arrival_end_ms - first_arrival_end_ms_);
    stamp.drift_ms_per_min =
        static_cast<double>(stamp.drift_ms) * 60000.0 / elapsed_ms;
    return stamp;
  }

 private:
  bool initialized_ = false;
  int reanchor_threshold_ms_ = 500;
  double alpha_ = 0.08;
  int max_correction_per_chunk_ms_ = 8;
  std::int64_t first_arrival_end_ms_ = 0;
  std::int64_t next_timestamp_ms_ = 0;
};

}  // namespace speaker_id
