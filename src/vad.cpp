#include "speaker_id/modules/vad.hpp"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace speaker_id {

struct SileroVadBackend::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "SileroVad"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;

  std::vector<float> h_state;
  std::vector<float> c_state;
  std::vector<float> context;
  float threshold = 0.25f;
  float neg_threshold = 0.175f;
  bool speech_active = false;
  int64_t speech_start_ms = 0;
  int64_t last_speech_time_ms = 0;
  int preroll_ms = 480;
  int min_silence_ms = 1200;
  int min_speech_ms = 450;
  int max_utterance_ms = 12000;

  std::vector<float> audio_buffer;
  int64_t total_samples_processed = 0;
  int64_t stream_start_ms = 0;

  Impl(const std::string& model_path,
       float threshold_val,
       int preroll,
       int min_silence,
       int min_speech,
       int max_utterance)
      : threshold(threshold_val),
        neg_threshold(threshold_val * 0.7f),
        preroll_ms(preroll),
        min_silence_ms(min_silence),
        min_speech_ms(min_speech),
        max_utterance_ms(max_utterance) {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    h_state.assign(1 * 1 * 128, 0.0f);
    c_state.assign(1 * 1 * 128, 0.0f);
    context.assign(64, 0.0f);
  }
};

SileroVadBackend::SileroVadBackend(const std::string& model_path,
                                   float threshold,
                                   int preroll_ms,
                                   int min_silence_ms,
                                   int min_speech_ms,
                                   int max_utterance_ms)
    : impl_(std::make_unique<Impl>(model_path,
                                   threshold,
                                   preroll_ms,
                                   min_silence_ms,
                                   min_speech_ms,
                                   max_utterance_ms)) {}

SileroVadBackend::~SileroVadBackend() = default;

std::vector<VadEvent> SileroVadBackend::AcceptAudio(const AudioChunkEvent& chunk) {
  std::vector<VadEvent> events;
  if (chunk.data.empty()) {
    return events;
  }

  if (impl_->stream_start_ms == 0) {
    impl_->stream_start_ms = chunk.timestamp_ms;
  }

  const size_t hop_size = 512;
  const size_t context_size = 64;
  
  impl_->audio_buffer.insert(impl_->audio_buffer.end(), chunk.data.begin(), chunk.data.end());
  size_t offset = 0;
  
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  while (offset + hop_size <= impl_->audio_buffer.size()) {
    std::vector<float> x_input(context_size + hop_size);
    std::copy(impl_->context.begin(), impl_->context.end(), x_input.begin());
    std::copy(impl_->audio_buffer.begin() + offset, impl_->audio_buffer.begin() + offset + hop_size, x_input.begin() + context_size);

    std::copy(impl_->audio_buffer.begin() + offset + hop_size - context_size, impl_->audio_buffer.begin() + offset + hop_size, impl_->context.begin());

    std::vector<int64_t> x_shape = {1, static_cast<int64_t>(context_size + hop_size)};
    std::vector<int64_t> state_shape = {1, 1, 128};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, x_input.data(), x_input.size(), x_shape.data(), x_shape.size());
    Ort::Value h_tensor = Ort::Value::CreateTensor<float>(memory_info, impl_->h_state.data(), impl_->h_state.size(), state_shape.data(), state_shape.size());
    Ort::Value c_tensor = Ort::Value::CreateTensor<float>(memory_info, impl_->c_state.data(), impl_->c_state.size(), state_shape.data(), state_shape.size());

    const char* input_names[] = {"input", "h", "c"};
    const char* output_names[] = {"speech_probs", "hn", "cn"};
    
    Ort::Value inputs[] = {std::move(input_tensor), std::move(h_tensor), std::move(c_tensor)};
    
    auto output_tensors = impl_->session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        inputs,
        3,
        output_names,
        3
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    float speech_prob = output_data[0];

    // Diagnostic logging for Silero VAD speech probability
    static int vad_eval_count = 0;
    ++vad_eval_count;
    if (vad_eval_count % 250 == 0 ||
        (!impl_->speech_active && speech_prob >= impl_->threshold)) {
      std::cout << "[SileroVAD] Eval " << vad_eval_count << ", Prob: " << speech_prob << " (Threshold: " << impl_->threshold << ")\n";
    }

    float* hn_data = output_tensors[1].GetTensorMutableData<float>();
    float* cn_data = output_tensors[2].GetTensorMutableData<float>();

    std::copy(hn_data, hn_data + impl_->h_state.size(), impl_->h_state.begin());
    std::copy(cn_data, cn_data + impl_->c_state.size(), impl_->c_state.begin());

    impl_->total_samples_processed += hop_size;
    int64_t current_time_ms = impl_->stream_start_ms + static_cast<int64_t>(impl_->total_samples_processed * 1000.0f / chunk.sample_rate);

    if (speech_prob >= impl_->threshold) {
      if (!impl_->speech_active) {
        impl_->speech_active = true;
        impl_->speech_start_ms = std::max(static_cast<int64_t>(0), current_time_ms - impl_->preroll_ms);
      }
      impl_->last_speech_time_ms = current_time_ms;
    }

    if (impl_->speech_active) {
      bool force_flush = (current_time_ms - impl_->speech_start_ms >= impl_->max_utterance_ms);
      bool silence_reached = (speech_prob < impl_->neg_threshold && (current_time_ms - impl_->last_speech_time_ms >= impl_->min_silence_ms));

      if (force_flush || silence_reached) {
        int64_t duration_ms = current_time_ms - impl_->speech_start_ms;
        if (duration_ms >= impl_->min_speech_ms) {
          VadEvent event;
          event.start_ms = impl_->speech_start_ms;
          event.end_ms = current_time_ms;
          event.confidence = speech_prob;
          events.push_back(event);
        }
        impl_->speech_active = false;
      }
    }

    offset += hop_size;
  }

  if (offset > 0) {
    impl_->audio_buffer.erase(impl_->audio_buffer.begin(), impl_->audio_buffer.begin() + offset);
  }

  return events;
}

bool SileroVadBackend::IsSpeechActive() const {
  return impl_->speech_active;
}

} // namespace speaker_id
