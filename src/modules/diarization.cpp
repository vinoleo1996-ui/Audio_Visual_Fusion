#include "speaker_id/modules/diarization.hpp"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace speaker_id {

namespace {

// Helper to normalize embedding
std::vector<float> NormalizeEmbedding(const std::vector<float>& vec) {
  if (vec.empty()) return vec;
  double sum_sq = 0.0;
  for (float val : vec) {
    sum_sq += static_cast<double>(val) * val;
  }
  float norm = static_cast<float>(std::sqrt(sum_sq));
  if (norm <= 0.0F) return vec;
  std::vector<float> normalized = vec;
  for (float& val : normalized) {
    val /= norm;
  }
  return normalized;
}

float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0F;
  float dot = 0.0F;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

// C++ implementation of SpeakerClusterer
class SpeakerClusterer {
 public:
  float threshold;
  float create_threshold;
  float merge_threshold;
  float ambiguous_margin = 0.04F;
  float update_rate = 0.20F;
  std::vector<std::vector<float>> centroids;
  std::vector<int> counts;

  SpeakerClusterer(float thresh, float create_thresh = 0.68F, float merge_thresh = 0.30F)
      : threshold(thresh), create_threshold(create_thresh), merge_threshold(merge_thresh) {}

  std::string AssignEmbedding(const std::vector<float>& embedding, bool allow_create, bool update) {
    auto emb = NormalizeEmbedding(embedding);
    if (emb.empty()) {
      return "";
    }
    if (centroids.empty()) {
      if (!allow_create) {
        return "";
      }
      centroids.push_back(emb);
      counts.push_back(1);
      return "spk_1";
    }

    std::vector<float> distances;
    for (const auto& centroid : centroids) {
      distances.push_back(1.0F - CosineSimilarity(emb, centroid));
    }

    auto min_it = std::min_element(distances.begin(), distances.end());
    int best_idx = static_cast<int>(std::distance(distances.begin(), min_it));
    float best_distance = *min_it;

    std::vector<float> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    float second_distance = sorted_distances.size() > 1 ? sorted_distances[1] : 2.0F;
    float second_gap = second_distance - sorted_distances[0];

    if (best_distance > create_threshold) {
      if (!allow_create) {
        return "spk_" + std::to_string(best_idx + 1);
      }
      centroids.push_back(emb);
      counts.push_back(1);
      int new_idx = static_cast<int>(centroids.size()) - 1;
      int kept_idx = MergeSimilarClusters(new_idx);
      return "spk_" + std::to_string(kept_idx + 1);
    }

    if (second_gap < ambiguous_margin || best_distance > threshold) {
      return "spk_" + std::to_string(best_idx + 1);
    }

    int count = counts[best_idx];
    float rate = std::min(std::max(update_rate, 0.01F), 1.0F / std::max(1, count));
    if (update) {
      std::vector<float> updated(emb.size());
      for (size_t i = 0; i < emb.size(); ++i) {
        updated[i] = centroids[best_idx][i] * (1.0F - rate) + emb[i] * rate;
      }
      centroids[best_idx] = NormalizeEmbedding(updated);
      counts[best_idx] = count + 1;
      best_idx = MergeSimilarClusters(best_idx);
    }
    return "spk_" + std::to_string(best_idx + 1);
  }

  int MergeSimilarClusters(int idx) {
    if (idx < 0 || idx >= static_cast<int>(centroids.size()) || centroids.size() < 2) {
      return idx;
    }
    const auto& target = centroids[idx];
    for (size_t other_idx = 0; other_idx < centroids.size(); ++other_idx) {
      if (static_cast<int>(other_idx) == idx) {
        continue;
      }
      float distance = 1.0F - CosineSimilarity(target, centroids[other_idx]);
      if (distance > merge_threshold) {
        continue;
      }
      int keep_idx = std::min(idx, static_cast<int>(other_idx));
      int drop_idx = std::max(idx, static_cast<int>(other_idx));

      int keep_count = counts[keep_idx];
      int drop_count = counts[drop_idx];
      std::vector<float> merged(target.size());
      for (size_t i = 0; i < target.size(); ++i) {
        merged[i] = centroids[keep_idx][i] * keep_count + centroids[drop_idx][i] * drop_count;
      }
      centroids[keep_idx] = NormalizeEmbedding(merged);
      counts[keep_idx] = keep_count + drop_count;

      centroids.erase(centroids.begin() + drop_idx);
      counts.erase(counts.begin() + drop_idx);
      return keep_idx;
    }
    return idx;
  }
};

// C++ RAII wrapper for Sherpa ONNX Speaker Embedding Extractor using C API
class SpeakerEmbeddingExtractor {
 public:
  SpeakerEmbeddingExtractor(const std::string& model_path) {
    SherpaOnnxSpeakerEmbeddingExtractorConfig config;
    config.model = model_path.c_str();
    config.num_threads = 1;
    config.provider = "cpu";
    config.debug = 0;
    p_ = SherpaOnnxCreateSpeakerEmbeddingExtractor(&config);
    if (!p_) {
      throw std::runtime_error("failed to create speaker embedding extractor: " + model_path);
    }
  }

  ~SpeakerEmbeddingExtractor() {
    if (p_) {
      SherpaOnnxDestroySpeakerEmbeddingExtractor(p_);
    }
  }

  int Dim() const {
    return SherpaOnnxSpeakerEmbeddingExtractorDim(p_);
  }

  std::vector<float> Extract(const std::vector<float>& samples, int sample_rate) {
    if (samples.size() < static_cast<size_t>(sample_rate * 0.45)) {
      return {};
    }
    const auto* stream = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(p_);
    if (!stream) {
      return {};
    }
    SherpaOnnxOnlineStreamAcceptWaveform(stream, sample_rate, samples.data(), static_cast<int32_t>(samples.size()));
    SherpaOnnxOnlineStreamInputFinished(stream);
    if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(p_, stream)) {
      SherpaOnnxDestroyOnlineStream(stream);
      return {};
    }
    const float* embedding = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(p_, stream);
    std::vector<float> result(embedding, embedding + Dim());
    SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(embedding);
    SherpaOnnxDestroyOnlineStream(stream);
    return NormalizeEmbedding(result);
  }

 private:
  const SherpaOnnxSpeakerEmbeddingExtractor* p_ = nullptr;
};

// Powerset definition
const int kMaxSpeakersPerChunk = 3;

// Powerset mappings
const std::vector<std::vector<int>> kPowersetClasses = {
    {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}
};

} // namespace

struct OrtDiarizationBackend::Impl {
  Ort::Env env;
  Ort::Session pyannote_session;
  std::unique_ptr<SpeakerEmbeddingExtractor> extractor;
  SpeakerClusterer clusterer;
  
  // Voice-face affinity probability matrix: voice_spk_id -> (person_track_id -> score)
  std::unordered_map<std::string, std::unordered_map<int, float>> affinity_matrix;
  float affinity_decay = 0.92F;
  float affinity_threshold = 0.68F;
  float affinity_margin = 0.12F;
  float segmentation_threshold = 0.45F;
  int min_segment_ms = 200;
  std::mutex mu;
  std::string input_name;
  std::string output_name;

  Impl(const std::string& pyannote_model_path,
       const std::string& cam_model_path,
       float seg_threshold,
       int min_segment_duration_ms)
      : env(ORT_LOGGING_LEVEL_WARNING, "Diarization"),
        pyannote_session(env, pyannote_model_path.c_str(), Ort::SessionOptions{}),
        extractor(std::make_unique<SpeakerEmbeddingExtractor>(cam_model_path)),
        clusterer(0.50F),
        segmentation_threshold(seg_threshold),
        min_segment_ms(min_segment_duration_ms) {
    Ort::AllocatorWithDefaultOptions allocator;
    input_name = pyannote_session.GetInputNameAllocated(0, allocator).get();
    output_name = pyannote_session.GetOutputNameAllocated(0, allocator).get();
  }

  // Softmax
  std::vector<float> Softmax(const float* logits, int size) {
    std::vector<float> probs(size);
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0F;
    for (int i = 0; i < size; ++i) {
      probs[i] = std::exp(logits[i] - max_val);
      sum += probs[i];
    }
    for (int i = 0; i < size; ++i) {
      probs[i] /= sum;
    }
    return probs;
  }
};

OrtDiarizationBackend::OrtDiarizationBackend(const std::string& pyannote_model_path,
                                             const std::string& cam_model_path,
                                             float segmentation_threshold,
                                             int min_segment_ms)
    : impl_(std::make_unique<Impl>(pyannote_model_path, cam_model_path, segmentation_threshold, min_segment_ms)) {}

OrtDiarizationBackend::~OrtDiarizationBackend() = default;

std::vector<DiarizationSegment> OrtDiarizationBackend::ProcessUtterance(const std::vector<float>& samples, int sample_rate) {
  if (samples.empty()) return {};
  std::lock_guard<std::mutex> lock(impl_->mu);

  const int target_samples = 160000; // 10s at 16kHz
  std::vector<DiarizationSegment> raw_segments;
  for (int chunk_start = 0; chunk_start < static_cast<int>(samples.size());
       chunk_start += target_samples) {
    const int chunk_samples =
        std::min(target_samples, static_cast<int>(samples.size()) - chunk_start);
    std::vector<float> padded(target_samples, 0.0F);
    std::copy_n(samples.begin() + chunk_start, chunk_samples, padded.begin());

    std::vector<int64_t> input_shape = {1, 1, target_samples};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, padded.data(), target_samples, input_shape.data(), input_shape.size());
    const char* input_names[] = {impl_->input_name.c_str()};
    const char* output_names[] = {impl_->output_name.c_str()};
    auto outputs = impl_->pyannote_session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* logits_data = outputs[0].GetTensorMutableData<float>();
    const auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 3 || output_shape[1] <= 0 ||
        output_shape[2] != static_cast<std::int64_t>(kPowersetClasses.size())) {
      throw std::runtime_error("unexpected pyannote segmentation output shape");
    }
    const int frames_per_chunk = static_cast<int>(output_shape[1]);
    const float frame_step_s =
        static_cast<float>(target_samples) / sample_rate / frames_per_chunk;

    const int valid_frames =
        std::min(frames_per_chunk,
                 std::max(1, static_cast<int>(
                                 std::ceil(chunk_samples /
                                           (frame_step_s * sample_rate)))));
    std::vector<std::vector<float>> spk_probs(
        valid_frames, std::vector<float>(kMaxSpeakersPerChunk, 0.0F));
    for (int frame = 0; frame < valid_frames; ++frame) {
      auto frame_probs = impl_->Softmax(logits_data + frame * 7, 7);
      for (int class_index = 0; class_index < 7; ++class_index) {
        for (int speaker : kPowersetClasses[class_index]) {
          spk_probs[frame][speaker] += frame_probs[class_index];
        }
      }
    }

    auto append_segment = [&](const std::vector<int>& active, int first_frame,
                              int last_frame) {
      if (active.empty()) {
        return;
      }
      DiarizationSegment segment;
      segment.left_sample =
          chunk_start + static_cast<int>(first_frame * frame_step_s * sample_rate);
      segment.right_sample = std::min(
          chunk_start + chunk_samples,
          chunk_start + static_cast<int>(last_frame * frame_step_s * sample_rate));
      if (segment.right_sample - segment.left_sample <
          impl_->min_segment_ms * sample_rate / 1000) {
        return;
      }
      segment.is_overlap = active.size() >= 2;
      for (int speaker : active) {
        segment.overlap_speakers.push_back("local_" + std::to_string(speaker + 1));
      }
      raw_segments.push_back(std::move(segment));
    };

    std::vector<int> current_active;
    int segment_start_frame = 0;
    for (int frame = 0; frame < valid_frames; ++frame) {
      std::vector<int> active;
      for (int speaker = 0; speaker < kMaxSpeakersPerChunk; ++speaker) {
        if (spk_probs[frame][speaker] > impl_->segmentation_threshold) {
          active.push_back(speaker);
        }
      }
      if (active != current_active) {
        append_segment(current_active, segment_start_frame, frame);
        current_active = std::move(active);
        segment_start_frame = frame;
      }
    }
    append_segment(current_active, segment_start_frame, valid_frames);
  }

  if (raw_segments.empty()) {
    // If no voice is detected, return full utterance as a single segment
    DiarizationSegment seg;
    seg.left_sample = 0;
    seg.right_sample = static_cast<int>(samples.size());
    seg.voice_spk_id = "";
    seg.is_overlap = false;
    return {seg};
  }

  // First Pass: Extract CAM++ embeddings from clean non-overlap segments and match them
  std::unordered_map<int, std::string> local_to_global;
  for (size_t i = 0; i < raw_segments.size(); ++i) {
    auto& seg = raw_segments[i];
    if (seg.is_overlap) continue;
    
    int left = std::max(0, seg.left_sample);
    int right = std::min(static_cast<int>(samples.size()), seg.right_sample);
    if (right <= left) continue;

    std::vector<float> seg_audio(samples.begin() + left, samples.begin() + right);
    auto emb = impl_->extractor->Extract(seg_audio, sample_rate);
    if (!emb.empty()) {
      std::string global_spk = impl_->clusterer.AssignEmbedding(emb, true, true);
      if (!global_spk.empty() && seg.overlap_speakers.size() == 1) {
        int local_idx = std::stoi(seg.overlap_speakers[0].substr(6)) - 1;
        local_to_global[local_idx] = global_spk;
        seg.voice_spk_id = global_spk;
      }
    }
  }

  // Second Pass: Assign global speaker IDs to overlap segments
  for (auto& seg : raw_segments) {
    if (seg.is_overlap) {
      std::vector<std::string> globals;
      for (const auto& local : seg.overlap_speakers) {
        int local_idx = std::stoi(local.substr(6)) - 1;
        if (local_to_global.count(local_idx) > 0) {
          globals.push_back(local_to_global[local_idx]);
        } else {
          // Attempt to assign embedding without creating new clusters
          int left = std::max(0, seg.left_sample);
          int right = std::min(static_cast<int>(samples.size()), seg.right_sample);
          if (right > left) {
            std::vector<float> seg_audio(samples.begin() + left, samples.begin() + right);
            auto emb = impl_->extractor->Extract(seg_audio, sample_rate);
            if (!emb.empty()) {
              std::string gspk = impl_->clusterer.AssignEmbedding(emb, false, false);
              if (!gspk.empty()) {
                globals.push_back(gspk);
                local_to_global[local_idx] = gspk;
              }
            }
          }
        }
      }
      // Deduplicate globals
      std::vector<std::string> unique_globals;
      std::unordered_set<std::string> seen;
      for (const auto& g : globals) {
        if (seen.insert(g).second) {
          unique_globals.push_back(g);
        }
      }
      if (!unique_globals.empty()) {
        std::string voice_spk;
        for (size_t j = 0; j < unique_globals.size(); ++j) {
          if (j > 0) voice_spk += "+";
          voice_spk += unique_globals[j];
        }
        seg.voice_spk_id = voice_spk;
      }
    }
  }

  // Adjacent segments merger: Unify contiguous segments of the same speaker
  std::vector<DiarizationSegment> merged;
  for (const auto& seg : raw_segments) {
    if (merged.empty()) {
      merged.push_back(seg);
      continue;
    }
    auto& prev = merged.back();
    if (prev.voice_spk_id == seg.voice_spk_id && prev.is_overlap == seg.is_overlap) {
      prev.right_sample = seg.right_sample;
      if (prev.is_overlap) {
        std::unordered_set<std::string> unique_spks(prev.overlap_speakers.begin(), prev.overlap_speakers.end());
        unique_spks.insert(seg.overlap_speakers.begin(), seg.overlap_speakers.end());
        prev.overlap_speakers = std::vector<std::string>(unique_spks.begin(), unique_spks.end());
      }
    } else {
      merged.push_back(seg);
    }
  }

  return merged;
}

std::string OrtDiarizationBackend::ResolveSpeaker(const std::string& voice_spk_id, const std::vector<TrackEvent>& visible_tracks) {
  if (voice_spk_id.empty()) return "ambiguous";
  std::lock_guard<std::mutex> lock(impl_->mu);

  // Check voice face affinity mapping
  if (impl_->affinity_matrix.count(voice_spk_id) == 0) {
    return voice_spk_id;
  }
  
  const auto& row = impl_->affinity_matrix[voice_spk_id];
  if (row.empty()) {
    return voice_spk_id;
  }

  // Sort and find best match in affinity
  std::vector<std::pair<int, float>> sorted;
  for (const auto& item : row) {
    sorted.push_back(item);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });

  int best_track_id = sorted[0].first;
  float best_score = sorted[0].second;
  float second_score = sorted.size() > 1 ? sorted[1].second : 0.0F;

  if (best_score >= impl_->affinity_threshold && best_score - second_score >= impl_->affinity_margin) {
    // If the track is currently visible, attribute to it
    for (const auto& track : visible_tracks) {
      if (track.person_track_id == best_track_id) {
        if (track.face_track_id) {
          return "I" + std::to_string(*track.face_track_id);
        }
        return "P" + std::to_string(track.person_track_id);
      }
    }
  }

  return voice_spk_id;
}

void OrtDiarizationBackend::UpdateAffinity(const std::string& voice_spk_id, int person_track_id, float evidence, const std::vector<TrackEvent>& visible_tracks) {
  if (voice_spk_id.empty() || person_track_id < 0) return;

  // Handle overlap split spk+spk
  if (voice_spk_id.find('+') != std::string::npos) {
    size_t start = 0;
    size_t end = voice_spk_id.find('+');
    while (end != std::string::npos) {
      UpdateAffinity(voice_spk_id.substr(start, end - start), person_track_id, evidence, visible_tracks);
      start = end + 1;
      end = voice_spk_id.find('+', start);
    }
    UpdateAffinity(voice_spk_id.substr(start), person_track_id, evidence, visible_tracks);
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mu);

  // Decay existing affinities
  auto& row = impl_->affinity_matrix[voice_spk_id];
  for (auto it = row.begin(); it != row.end(); ) {
    it->second *= impl_->affinity_decay;
    if (it->second < 0.05F) {
      it = row.erase(it);
    } else {
      ++it;
    }
  }

  // Update with new evidence
  float capped_evidence = std::max(0.55F, std::min(0.95F, evidence));
  float prev = row[person_track_id];
  row[person_track_id] = prev + (1.0F - prev) * capped_evidence * 0.45F;

  // Competitive decay: Apply 0.70x factor to all other currently visible tracks to enforce mutual exclusion
  for (const auto& track : visible_tracks) {
    if (track.person_track_id != person_track_id) {
      if (row.count(track.person_track_id) > 0) {
        row[track.person_track_id] *= 0.70F;
        if (row[track.person_track_id] < 0.05F) {
          row.erase(track.person_track_id);
        }
      }
    }
  }
}

} // namespace speaker_id
