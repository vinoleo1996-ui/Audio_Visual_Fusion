#include "speaker_id/modules/asd.hpp"

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace speaker_id {
namespace {

constexpr int kMfccFrames = 100;
constexpr int kMfccCoefficients = 13;
constexpr int kFilterBanks = 26;
constexpr int kVisualFrames = 25;
constexpr int kVisualWidth = 112;
constexpr int kVisualPixels = kVisualWidth * kVisualWidth;
constexpr float kPi = 3.14159265358979323846F;

float HertzToMel(float hz) {
  return 2595.0F * std::log10(1.0F + hz / 700.0F);
}

float MelToHertz(float mel) {
  return 700.0F * (std::pow(10.0F, mel / 2595.0F) - 1.0F);
}

std::vector<float> ExtractMfcc(const std::vector<float>& samples, int sample_rate) {
  if (samples.empty() || sample_rate <= 0) {
    return {};
  }

  std::vector<float> normalized(samples);
  const float square_sum =
      std::inner_product(normalized.begin(), normalized.end(), normalized.begin(), 0.0F);
  const float rms = std::sqrt(square_sum / static_cast<float>(normalized.size()));
  if (rms > 1e-4F) {
    const float scale = 0.12F / rms;
    for (auto& value : normalized) {
      value = std::clamp(value * scale, -1.0F, 32767.0F / 32768.0F);
    }
  }

  std::vector<float> emphasized(normalized.size());
  emphasized[0] = normalized[0] * 32768.0F;
  for (std::size_t index = 1; index < normalized.size(); ++index) {
    emphasized[index] = (normalized[index] - 0.97F * normalized[index - 1]) * 32768.0F;
  }

  const int frame_length = static_cast<int>(std::round(0.025F * sample_rate));
  const int frame_step = static_cast<int>(std::round(0.010F * sample_rate));
  const int fft_size = 512;
  const int spectrum_size = fft_size / 2 + 1;
  const int frame_count =
      emphasized.size() <= static_cast<std::size_t>(frame_length)
          ? 1
          : 1 + static_cast<int>(std::ceil(
                    static_cast<float>(emphasized.size() - frame_length) / frame_step));

  const float low_mel = HertzToMel(0.0F);
  const float high_mel = HertzToMel(static_cast<float>(sample_rate) / 2.0F);
  std::vector<int> bins(kFilterBanks + 2);
  for (int index = 0; index < kFilterBanks + 2; ++index) {
    const float mel =
        low_mel + (high_mel - low_mel) * static_cast<float>(index) /
                      static_cast<float>(kFilterBanks + 1);
    bins[index] = static_cast<int>(
        std::floor((fft_size + 1) * MelToHertz(mel) / static_cast<float>(sample_rate)));
  }

  std::vector<float> output;
  output.reserve(kMfccFrames * kMfccCoefficients);
  std::vector<float> last_coefficients(kMfccCoefficients, 0.0F);
  for (int frame_index = 0; frame_index < std::min(frame_count, kMfccFrames); ++frame_index) {
    cv::Mat signal = cv::Mat::zeros(1, fft_size, CV_32F);
    float energy = 0.0F;
    for (int sample_index = 0; sample_index < frame_length; ++sample_index) {
      const std::size_t source_index =
          static_cast<std::size_t>(frame_index * frame_step + sample_index);
      const float value = source_index < emphasized.size() ? emphasized[source_index] : 0.0F;
      signal.at<float>(0, sample_index) = value;
      energy += value * value;
    }
    energy = std::max(energy, std::numeric_limits<float>::epsilon());

    cv::Mat spectrum;
    cv::dft(signal, spectrum, cv::DFT_COMPLEX_OUTPUT);
    std::vector<float> power(spectrum_size, 0.0F);
    for (int index = 0; index < spectrum_size; ++index) {
      const auto value = spectrum.at<cv::Vec2f>(0, index);
      power[index] = (value[0] * value[0] + value[1] * value[1]) /
                     static_cast<float>(fft_size);
    }

    std::vector<float> log_filter_banks(kFilterBanks, 0.0F);
    for (int filter = 0; filter < kFilterBanks; ++filter) {
      float value = 0.0F;
      const int left = bins[filter];
      const int center = std::max(left + 1, bins[filter + 1]);
      const int right = std::max(center + 1, bins[filter + 2]);
      for (int index = left; index < center && index < spectrum_size; ++index) {
        value += power[index] * static_cast<float>(index - left) /
                 static_cast<float>(center - left);
      }
      for (int index = center; index < right && index < spectrum_size; ++index) {
        value += power[index] * static_cast<float>(right - index) /
                 static_cast<float>(right - center);
      }
      log_filter_banks[filter] =
          std::log(std::max(value, std::numeric_limits<float>::epsilon()));
    }

    for (int coefficient = 0; coefficient < kMfccCoefficients; ++coefficient) {
      float value = 0.0F;
      for (int filter = 0; filter < kFilterBanks; ++filter) {
        value += log_filter_banks[filter] *
                 std::cos(kPi * coefficient * (static_cast<float>(filter) + 0.5F) /
                          static_cast<float>(kFilterBanks));
      }
      value *= coefficient == 0 ? std::sqrt(1.0F / kFilterBanks)
                                : std::sqrt(2.0F / kFilterBanks);
      last_coefficients[coefficient] = value;
    }
    last_coefficients[0] = std::log(energy);
    output.insert(output.end(), last_coefficients.begin(), last_coefficients.end());
  }

  if (output.empty()) {
    return {};
  }
  while (output.size() < static_cast<std::size_t>(kMfccFrames * kMfccCoefficients)) {
    output.insert(output.end(), last_coefficients.begin(), last_coefficients.end());
  }
  output.resize(kMfccFrames * kMfccCoefficients);
  return output;
}

std::vector<float> SelectVisualFrames(
    const std::vector<AsdVisualSample>& samples,
    std::int64_t start_ms,
    std::int64_t end_ms) {
  if (samples.empty() || end_ms <= start_ms) {
    return {};
  }
  const auto duration_ms = end_ms - start_ms;
  const auto max_gap_ms = std::max<std::int64_t>(250, duration_ms * 3 / kVisualFrames);
  std::vector<float> visual;
  visual.reserve(kVisualFrames * kVisualPixels);
  for (int frame_index = 0; frame_index < kVisualFrames; ++frame_index) {
    const auto target_ms =
        start_ms + duration_ms * frame_index / std::max(1, kVisualFrames - 1);
    const auto nearest = std::min_element(
        samples.begin(), samples.end(), [target_ms](const auto& left, const auto& right) {
          return std::llabs(left.timestamp_ms - target_ms) <
                 std::llabs(right.timestamp_ms - target_ms);
        });
    if (nearest == samples.end() ||
        std::llabs(nearest->timestamp_ms - target_ms) > max_gap_ms ||
        nearest->grayscale_112.size() != kVisualPixels) {
      return {};
    }
    visual.insert(visual.end(), nearest->grayscale_112.begin(), nearest->grayscale_112.end());
  }
  return visual;
}

float Mean(const float* values, std::size_t count) {
  if (count == 0) {
    return 0.0F;
  }
  return std::accumulate(values, values + count, 0.0F) / static_cast<float>(count);
}

}  // namespace

std::vector<ActiveSpeakerScore> SimpleAsdBackend::ScoreWindow(const AsdInputWindow& window) {
  std::vector<ActiveSpeakerScore> scores;
  for (const auto& track : window.tracks) {
    ActiveSpeakerScore score;
    score.person_track_id = track.person_track_id;
    score.face_track_id = track.face_track_id;
    score.timestamp_ms = window.end_ms;
    score.p_active = window.vad_segments.empty() ? 0.15F : 0.78F;
    score.p_audible_speaking = score.p_active;
    score.p_visual_mouth_motion = score.p_active;
    score.p_av_sync = score.p_active;
    score.face_quality = (track.quality == Quality::kGood) ? 0.9F : 0.5F;
    score.uncertainty_flags.push_back("synthetic_test_backend");
    scores.push_back(score);
  }
  return scores;
}

struct OrtLrAsdBackend::Impl {
  explicit Impl(const std::string& model_path) {
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), options);
  }

  float Smooth(int track_id, float raw) {
    const auto previous = probabilities.find(track_id);
    if (previous == probabilities.end()) {
      probabilities[track_id] = raw;
      active[track_id] = raw >= 0.55F;
      return raw;
    }
    const float fused = raw >= previous->second
                            ? 0.35F * previous->second + 0.65F * raw
                            : 0.70F * previous->second + 0.30F * raw;
    const bool was_active = active[track_id];
    active[track_id] = fused >= (was_active ? 0.35F : 0.55F);
    probabilities[track_id] = fused;
    return fused;
  }

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "LrAsd"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  std::mutex mu;
  std::map<int, float> probabilities;
  std::map<int, bool> active;
};

OrtLrAsdBackend::OrtLrAsdBackend(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

OrtLrAsdBackend::~OrtLrAsdBackend() = default;

std::vector<ActiveSpeakerScore> OrtLrAsdBackend::ScoreWindow(const AsdInputWindow& window) {
  std::vector<ActiveSpeakerScore> scores;
  const auto mfcc = ExtractMfcc(window.audio_samples, window.sample_rate);
  if (mfcc.empty()) {
    return scores;
  }

  std::lock_guard<std::mutex> lock(impl_->mu);
  std::map<int, bool> visible_track_ids;
  for (const auto& track : window.tracks) {
    visible_track_ids[track.person_track_id] = true;
    const auto visual_it = window.visual_samples.find(track.person_track_id);
    if (visual_it == window.visual_samples.end()) {
      continue;
    }
    const auto visual = SelectVisualFrames(visual_it->second, window.start_ms, window.end_ms);
    if (visual.empty()) {
      continue;
    }

    std::vector<int64_t> audio_shape = {1, kMfccFrames, kMfccCoefficients};
    std::vector<int64_t> visual_shape = {1, kVisualFrames, kVisualWidth, kVisualWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto audio_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(mfcc.data()), mfcc.size(), audio_shape.data(),
        audio_shape.size());
    auto visual_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(visual.data()), visual.size(), visual_shape.data(),
        visual_shape.size());
    const char* input_names[] = {"audio_mfcc", "visual_crops"};
    const char* output_names[] = {"p_active"};
    Ort::Value inputs[] = {std::move(audio_tensor), std::move(visual_tensor)};
    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, input_names, inputs, 2, output_names, 1);
    const auto shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    std::size_t score_count = 1;
    for (const auto dimension : shape) {
      score_count *= static_cast<std::size_t>(std::max<std::int64_t>(1, dimension));
    }
    const float raw = std::clamp(
        Mean(outputs.front().GetTensorData<float>(), score_count), 0.0F, 1.0F);

    ActiveSpeakerScore score;
    score.person_track_id = track.person_track_id;
    score.face_track_id = track.face_track_id;
    score.timestamp_ms = window.end_ms;
    score.p_active = impl_->Smooth(track.person_track_id, raw);
    score.p_audible_speaking = window.speech_active ? 1.0F : 0.0F;
    score.p_visual_mouth_motion = raw;
    score.p_av_sync = raw;
    score.face_quality = track.face.quality_score;
    if (!track.face.face_bbox_observed) {
      score.uncertainty_flags.push_back("projected_face_bbox");
    }
    scores.push_back(std::move(score));
  }

  for (auto iterator = impl_->probabilities.begin(); iterator != impl_->probabilities.end();) {
    if (visible_track_ids.count(iterator->first) == 0) {
      impl_->active.erase(iterator->first);
      iterator = impl_->probabilities.erase(iterator);
    } else {
      ++iterator;
    }
  }
  return scores;
}

}  // namespace speaker_id
