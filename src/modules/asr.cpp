#include "speaker_id/modules/asr.hpp"
#include "sherpa-onnx/c-api/cxx-api.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace speaker_id {

// Helper functions matching Python behavior
namespace {

std::string Trim(std::string text) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
  text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
  return text;
}

std::string NormalizeAsrText(const std::string& text) {
  std::string normalized;
  bool in_space = false;
  for (char ch : text) {
    if (ch == '\n') {
      ch = ' ';
    }
    if (std::isspace(static_cast<unsigned char>(ch))) {
      if (!in_space) {
        normalized.push_back(' ');
        in_space = true;
      }
    } else {
      normalized.push_back(ch);
      in_space = false;
    }
  }
  return Trim(normalized);
}

struct SignalStats {
  int samples = 0;
  float rms = 0.0F;
  float peak = 0.0F;
};

SignalStats ComputeSignalStats(const std::vector<float>& samples) {
  SignalStats stats;
  if (samples.empty()) {
    return stats;
  }
  stats.samples = static_cast<int>(samples.size());
  double sum_sq = 0.0;
  for (float val : samples) {
    float clean_val = std::isnan(val) || std::isinf(val) ? 0.0F : val;
    stats.peak = std::max(stats.peak, std::abs(clean_val));
    sum_sq += static_cast<double>(clean_val) * clean_val;
  }
  stats.rms = static_cast<float>(std::sqrt(sum_sq / samples.size()));
  return stats;
}

int TrailingLowEnergyMs(const std::vector<float>& samples, int sample_rate, float threshold) {
  if (samples.empty()) {
    return 0;
  }
  int frame = std::max(1, sample_rate * 2 / 100); // 20ms frame
  int trailing = 0;
  for (int end = static_cast<int>(samples.size()); end > 0; end -= frame) {
    int start = std::max(0, end - frame);
    float peak = 0.0F;
    for (int i = start; i < end; ++i) {
      peak = std::max(peak, std::abs(samples[i]));
    }
    if (peak > threshold) {
      break;
    }
    trailing += (end - start);
  }
  return static_cast<int>(std::round(static_cast<double>(trailing) * 1000.0 / sample_rate));
}

int LinguisticCharCount(const std::string& text) {
  int count = 0;
  for (char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch)) || (static_cast<unsigned char>(ch) >= 0x80)) {
      ++count; // Chinese/UTF-8 character or ascii alnum
    }
  }
  return count;
}

bool IsProbableHallucination(const std::string& text, int min_text_chars) {
  std::string stripped = NormalizeAsrText(text);
  if (stripped.empty()) {
    return true;
  }
  if (LinguisticCharCount(stripped) < min_text_chars) {
    return true;
  }
  const std::unordered_set<std::string> exact_boilerplate = {
      "字幕", "音乐", "[音乐]", "（音乐）"};
  if (exact_boilerplate.count(stripped) > 0) {
    return true;
  }
  const std::vector<std::string> boilerplate = {
      "字幕由", "本视频", "感谢观看", "谢谢观看", "谢谢您的观看",
      "请点赞", "请订阅", "请不吝点赞", "明镜与点点栏目", "下次再见"};
  for (const auto& item : boilerplate) {
    if (stripped.find(item) != std::string::npos) {
      return true;
    }
  }
  
  // Repeating characters filter
  std::string compact;
  for (char ch : stripped) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      compact.push_back(ch);
    }
  }
  if (compact.size() >= 6) {
    std::unordered_map<char, int> counts;
    int max_count = 0;
    for (char ch : compact) {
      max_count = std::max(max_count, ++counts[ch]);
    }
    if (static_cast<float>(max_count) / compact.size() > 0.55F) {
      return true;
    }
  }
  if (compact.size() >= 8) {
    std::unordered_set<char> unique_chars(compact.begin(), compact.end());
    if (static_cast<float>(unique_chars.size()) / compact.size() < 0.25F) {
      return true;
    }
  }
  return false;
}

void RequireReadableFile(const std::string& path, const std::string& label) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("missing " + label + ": " + path);
  }
}

float StableTokenRatio(
    const std::vector<std::string>& previous,
    const std::vector<std::string>& current) {
  if (current.empty()) {
    return 0.0F;
  }
  std::size_t stable = 0;
  while (stable < previous.size() && stable < current.size() &&
         previous[stable] == current[stable]) {
    ++stable;
  }
  return static_cast<float>(stable) / static_cast<float>(current.size());
}

} // namespace

struct SherpaAsrBackend::Impl {
  std::unique_ptr<sherpa_onnx::cxx::OfflineRecognizer> offline_recognizer;
  std::unique_ptr<sherpa_onnx::cxx::OnlineRecognizer> online_recognizer;
  std::unique_ptr<sherpa_onnx::cxx::OnlineStream> online_stream;
  AsrConfig config;
  std::string language;
  std::vector<std::string> previous_partial_tokens;

  std::mutex buffer_mu;
  std::vector<float> audio_buffer;
  int64_t buffer_start_time_ms = -1;
  
  Impl(const AsrConfig& asr_config, int cpu_threads)
      : config(asr_config), language(asr_config.language) {
    
    // 1. Initialize offline SenseVoice recognizer
    sherpa_onnx::cxx::OfflineRecognizerConfig off_config;
    off_config.feat_config.sample_rate = 16000;
    off_config.feat_config.feature_dim = 80;
    off_config.decoding_method = "greedy_search";
    
    std::string model_path = config.model_dir + "/model.int8.onnx";
    std::ifstream quantized_model(model_path);
    if (!quantized_model.good()) {
      model_path = config.model_dir + "/model.onnx";
    }
    RequireReadableFile(model_path, "SenseVoice model");
    RequireReadableFile(config.model_dir + "/tokens.txt", "SenseVoice tokens");
    off_config.model_config.sense_voice.model = model_path;
    off_config.model_config.sense_voice.language = language;
    off_config.model_config.sense_voice.use_itn = true;
    off_config.model_config.tokens = config.model_dir + "/tokens.txt";
    off_config.model_config.num_threads = cpu_threads;
    off_config.model_config.provider = "cpu";
    auto offline = sherpa_onnx::cxx::OfflineRecognizer::Create(off_config);
    if (!offline.Get()) {
      throw std::runtime_error("failed to initialize SenseVoice recognizer");
    }
    offline_recognizer =
        std::make_unique<sherpa_onnx::cxx::OfflineRecognizer>(std::move(offline));

    // 2. Initialize online Zipformer recognizer
    sherpa_onnx::cxx::OnlineRecognizerConfig on_config;
    on_config.feat_config.sample_rate = 16000;
    on_config.feat_config.feature_dim = 80;
    on_config.decoding_method = "greedy_search";
    on_config.model_config.zipformer2_ctc.model =
        config.streaming_model_dir + "/model.int8.onnx";
    on_config.model_config.tokens = config.streaming_model_dir + "/tokens.txt";
    on_config.model_config.num_threads = 1;
    on_config.model_config.provider = "cpu";
    RequireReadableFile(on_config.model_config.zipformer2_ctc.model, "Zipformer model");
    RequireReadableFile(on_config.model_config.tokens, "Zipformer tokens");
    auto online = sherpa_onnx::cxx::OnlineRecognizer::Create(on_config);
    if (!online.Get()) {
      throw std::runtime_error("failed to initialize Zipformer recognizer");
    }
    online_recognizer =
        std::make_unique<sherpa_onnx::cxx::OnlineRecognizer>(std::move(online));
  }
};

SherpaAsrBackend::SherpaAsrBackend(const AsrConfig& config, int cpu_threads)
    : impl_(std::make_unique<Impl>(config, cpu_threads)) {}

SherpaAsrBackend::~SherpaAsrBackend() = default;

AsrResult SherpaAsrBackend::Transcribe(const std::vector<float>& samples, int sample_rate) {
  AsrResult result;
  if (!impl_->offline_recognizer) {
    return result;
  }
  if (sample_rate != 16000) {
    return result;
  }
  
  // 1. Check signal threshold and length
  int min_samples = sample_rate * impl_->config.min_audio_ms / 1000;
  if (static_cast<int>(samples.size()) < min_samples) {
    return result;
  }
  
  auto stats = ComputeSignalStats(samples);
  const float silence_peak = impl_->config.silence_peak;
  if (stats.peak < silence_peak) {
    return result;
  }
  
  // 2. Add tail padding if needed
  std::vector<float> padded_samples = samples;
  const int tail_padding_ms = impl_->config.tail_padding_ms;
  const float tail_threshold = impl_->config.tail_silence_peak;
  int existing_tail_ms = TrailingLowEnergyMs(samples, sample_rate, tail_threshold);
  int padding_ms = std::max(0, tail_padding_ms - existing_tail_ms);
  if (padding_ms > 0) {
    int pad_samples = sample_rate * padding_ms / 1000;
    padded_samples.insert(padded_samples.end(), pad_samples, 0.0F);
  }
  
  auto stream = impl_->offline_recognizer->CreateStream();
  stream.AcceptWaveform(sample_rate, padded_samples.data(), static_cast<int32_t>(padded_samples.size()));
  impl_->offline_recognizer->Decode(&stream);
  auto native_result = impl_->offline_recognizer->GetResult(&stream);
  
  std::cout << "[SherpaASR] Decoded raw text: \"" << native_result.text << "\", samples: " << samples.size() << "\n";
  
  std::string text = NormalizeAsrText(native_result.text);
  result.language = native_result.lang.empty() ? impl_->language : native_result.lang;
  result.tokens = native_result.tokens;
  result.token_timestamps_s = native_result.timestamps;
  
  // 4. Hallucination filter
  if (IsProbableHallucination(text, impl_->config.min_text_chars)) {
    result.is_hallucination = true;
    return result;
  }
  
  result.text = text;
  result.confidence = text.empty() ? 0.0F : 0.80F;
  result.stability = text.empty() ? 0.0F : 1.0F;
  return result;
}

void SherpaAsrBackend::BeginStreaming() {
  if (impl_->online_recognizer) {
    impl_->previous_partial_tokens.clear();
    impl_->online_stream = std::make_unique<sherpa_onnx::cxx::OnlineStream>(
        impl_->online_recognizer->CreateStream());
  }
}

AsrResult SherpaAsrBackend::StreamAudio(const std::vector<float>& samples, int sample_rate) {
  AsrResult result;
  if (!impl_->online_recognizer || !impl_->online_stream) {
    return result;
  }
  
  impl_->online_stream->AcceptWaveform(sample_rate, samples.data(), static_cast<int32_t>(samples.size()));
  while (impl_->online_recognizer->IsReady(impl_->online_stream.get())) {
    impl_->online_recognizer->Decode(impl_->online_stream.get());
  }
  
  auto native_result = impl_->online_recognizer->GetResult(impl_->online_stream.get());
  result.text = NormalizeAsrText(native_result.text);
  result.tokens = native_result.tokens;
  result.token_timestamps_s = native_result.timestamps;
  result.stability = StableTokenRatio(impl_->previous_partial_tokens, result.tokens);
  result.confidence = 0.45F + 0.45F * result.stability;
  result.language = impl_->language;
  impl_->previous_partial_tokens = result.tokens;
  return result;
}

AsrResult SherpaAsrBackend::EndStreaming() {
  AsrResult result;
  if (impl_->online_recognizer && impl_->online_stream) {
    impl_->online_stream->InputFinished();
    while (impl_->online_recognizer->IsReady(impl_->online_stream.get())) {
      impl_->online_recognizer->Decode(impl_->online_stream.get());
    }
    auto native_result = impl_->online_recognizer->GetResult(impl_->online_stream.get());
    result.text = NormalizeAsrText(native_result.text);
    result.tokens = native_result.tokens;
    result.token_timestamps_s = native_result.timestamps;
    result.confidence = 0.70F;
    result.stability = result.text.empty() ? 0.0F : 1.0F;
    result.language = impl_->language;
    impl_->online_stream.reset();
    impl_->previous_partial_tokens.clear();
  }
  return result;
}

void SherpaAsrBackend::PushAudio(const AudioChunkEvent& chunk) {
  std::lock_guard<std::mutex> lock(impl_->buffer_mu);
  if (impl_->buffer_start_time_ms == -1) {
    impl_->buffer_start_time_ms = chunk.timestamp_ms;
  }
  impl_->audio_buffer.insert(impl_->audio_buffer.end(), chunk.data.begin(), chunk.data.end());
  
  const size_t max_samples = chunk.sample_rate * 25; // 25s buffer max
  if (impl_->audio_buffer.size() > max_samples) {
    size_t erase_len = impl_->audio_buffer.size() - max_samples;
    impl_->audio_buffer.erase(impl_->audio_buffer.begin(), impl_->audio_buffer.begin() + erase_len);
    impl_->buffer_start_time_ms += static_cast<int64_t>(erase_len * 1000.0 / chunk.sample_rate);
  }
}

std::vector<UtteranceEvent> SherpaAsrBackend::AcceptSpeechSegment(
    const VadEvent& seg,
    const std::vector<float>& speech_samples,
    DiarizationEngine* diarization) {
  if (speech_samples.empty()) {
    return {};
  }

  const std::vector<float>& samples = speech_samples;
  int sample_rate = 16000;

  // 1. Run full-text SenseVoice transcription first as reference and ultimate fallback
  AsrResult full_res = Transcribe(samples, sample_rate);
  if (full_res.text.empty() || full_res.is_hallucination) {
    std::cout << "[SherpaASR] Full utterance transcribes to empty/hallucination. Skipping segment.\n";
    return {};
  }

  std::vector<UtteranceEvent> results;

  // 2. Perform overlap-aware speaker segmentation if diarization is provided
  if (diarization) {
    auto dsegs = diarization->ProcessUtterance(samples, sample_rate);
    if (!dsegs.empty() && !(dsegs.size() == 1 && dsegs[0].voice_spk_id.empty())) {
      int idx = 1;
      std::string combined_diarized_text = "";
      std::vector<UtteranceEvent> temp_results;

      for (const auto& dseg : dsegs) {
        int left = std::max(0, dseg.left_sample);
        int right = std::min(static_cast<int>(samples.size()), dseg.right_sample);
        if (right <= left) continue;

        std::vector<float> sub_samples(samples.begin() + left, samples.begin() + right);
        AsrResult res = Transcribe(sub_samples, sample_rate);
        if (res.text.empty() || res.is_hallucination) {
          continue;
        }

        UtteranceEvent ev;
        ev.utterance_id = "utt_" + std::to_string(seg.start_ms) + "_" + std::to_string(idx++);
        ev.start_ms = seg.start_ms + static_cast<int64_t>(left * 1000.0 / sample_rate);
        ev.end_ms = seg.start_ms + static_cast<int64_t>(right * 1000.0 / sample_rate);
        ev.text = res.text;
        ev.final = true;
        ev.confidence = res.confidence;
        ev.stability = res.stability;
        ev.tokens = res.tokens;
        ev.token_timestamps_s = res.token_timestamps_s;
        ev.voice_spk_id = dseg.voice_spk_id;
        ev.is_overlap = dseg.is_overlap;
        temp_results.push_back(ev);

        combined_diarized_text += res.text;
      }

      // 3. Fallback logic: check if diarization splits caused severe transcription drop (due to very short segments)
      int full_char_count = LinguisticCharCount(full_res.text);
      int diarized_char_count = LinguisticCharCount(combined_diarized_text);

      if (temp_results.empty() || diarized_char_count < 0.75 * full_char_count) {
        std::cout << "[SherpaASR] Diarization sub-segments text length drops too much: "
                  << diarized_char_count << " vs " << full_char_count
                  << ". Falling back to high-quality full segment transcription.\n";
        
        // Find the dominant voice speaker in the diarization segments to attribute to the full VAD block
        std::string best_spk = "";
        int max_len = 0;
        bool has_overlap = false;
        for (const auto& dseg : dsegs) {
          int len = dseg.right_sample - dseg.left_sample;
          if (len > max_len && !dseg.voice_spk_id.empty()) {
            max_len = len;
            best_spk = dseg.voice_spk_id;
            has_overlap = dseg.is_overlap;
          }
        }
        
        UtteranceEvent ev;
        ev.utterance_id = "utt_" + std::to_string(seg.start_ms);
        ev.start_ms = seg.start_ms;
        ev.end_ms = seg.end_ms;
        ev.text = full_res.text;
        ev.final = true;
        ev.confidence = full_res.confidence;
        ev.stability = full_res.stability;
        ev.tokens = full_res.tokens;
        ev.token_timestamps_s = full_res.token_timestamps_s;
        ev.voice_spk_id = best_spk;
        ev.is_overlap = has_overlap;
        results.push_back(ev);
      } else {
        results = std::move(temp_results);
      }
    }
  }

  // 4. Fallback if no diarization was provided or results is empty
  if (results.empty()) {
    UtteranceEvent ev;
    ev.utterance_id = "utt_" + std::to_string(seg.start_ms);
    ev.start_ms = seg.start_ms;
    ev.end_ms = seg.end_ms;
    ev.text = full_res.text;
    ev.final = true;
    ev.confidence = full_res.confidence;
    ev.stability = full_res.stability;
    ev.tokens = full_res.tokens;
    ev.token_timestamps_s = full_res.token_timestamps_s;
    results.push_back(ev);
  }

  return results;
}

} // namespace speaker_id
