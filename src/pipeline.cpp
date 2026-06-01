#include "speaker_id/core/pipeline.hpp"

#include <chrono>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace speaker_id {
namespace {

std::int64_t MonotonicNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double Percentile(std::deque<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::round((values.size() - 1) * std::clamp(percentile, 0.0, 1.0)));
  return values[index];
}

std::string EscapeJsonString(const std::string& input) {
  std::ostringstream output;
  for (const char ch : input) {
    switch (ch) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << ch; break;
    }
  }
  return output.str();
}

bool IsFiniteBox(const BBox& box) {
  return std::isfinite(box.x1) && std::isfinite(box.y1) &&
         std::isfinite(box.x2) && std::isfinite(box.y2);
}

bool ClampBox(BBox& box, int width, int height, int min_size_px) {
  if (!IsFiniteBox(box) || width <= 0 || height <= 0) {
    return false;
  }
  box.x1 = std::clamp(box.x1, 0.0F, static_cast<float>(width));
  box.y1 = std::clamp(box.y1, 0.0F, static_cast<float>(height));
  box.x2 = std::clamp(box.x2, 0.0F, static_cast<float>(width));
  box.y2 = std::clamp(box.y2, 0.0F, static_cast<float>(height));
  return box.x2 - box.x1 >= min_size_px && box.y2 - box.y1 >= min_size_px;
}

bool HasRenderableFace(const TrackEvent& track) {
  return track.face.face_bbox_last_observed_ms > 0 &&
         track.face.observation_state != "expired";
}

int MaxCropGapMs(const std::vector<AsdVisualSample>& samples) {
  int max_gap_ms = 0;
  for (std::size_t index = 1; index < samples.size(); ++index) {
    max_gap_ms = std::max(
        max_gap_ms,
        static_cast<int>(samples[index].timestamp_ms - samples[index - 1].timestamp_ms));
  }
  return max_gap_ms;
}

std::uint64_t HashFloats(const std::vector<float>& values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const float value : values) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    for (int shift = 0; shift < 32; shift += 8) {
      hash ^= static_cast<std::uint8_t>((bits >> shift) & 0xFF);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

std::vector<std::int64_t> SelectVisualTimestamps(
    const std::vector<AsdVisualSample>& samples, std::int64_t start_ms,
    std::int64_t end_ms) {
  constexpr int kVisualFrames = 25;
  std::vector<std::int64_t> timestamps;
  if (samples.empty() || end_ms <= start_ms) {
    return timestamps;
  }
  const auto duration_ms = end_ms - start_ms;
  const auto max_gap_ms = std::max<std::int64_t>(250, duration_ms * 3 / kVisualFrames);
  for (int frame_index = 0; frame_index < kVisualFrames; ++frame_index) {
    const auto target_ms =
        start_ms + duration_ms * frame_index / std::max(1, kVisualFrames - 1);
    const auto nearest = std::min_element(
        samples.begin(), samples.end(), [target_ms](const auto& left, const auto& right) {
          return std::llabs(left.timestamp_ms - target_ms) <
                 std::llabs(right.timestamp_ms - target_ms);
        });
    if (nearest == samples.end() ||
        std::llabs(nearest->timestamp_ms - target_ms) > max_gap_ms) {
      return {};
    }
    timestamps.push_back(nearest->timestamp_ms);
  }
  return timestamps;
}

void WritePgm(const std::filesystem::path& path, const std::vector<float>& grayscale_112) {
  std::ofstream crop(path, std::ios::binary);
  crop << "P5\n112 112\n255\n";
  for (const auto pixel : grayscale_112) {
    crop.put(static_cast<char>(
        std::clamp(static_cast<int>(std::lround(pixel)), 0, 255)));
  }
}

FusionTrackView MakeTrackView(const TrackEvent& track, float p_active = 0.0F) {
  FusionTrackView view;
  view.person_track_id = track.person_track_id;
  view.track_snapshot_sequence = track.track_snapshot_sequence;
  view.snapshot_timestamp_ms = track.snapshot_timestamp_ms;
  view.face_track_id = track.face_track_id;
  view.bbox = track.bbox;
  view.quality = track.quality;
  view.p_active = p_active;
  view.face = track.face;
  view.body = track.body;
  view.render = track.render;
  view.identity_id = track.identity_id;
  view.identity_name = track.identity_name;
  view.identity_state = track.identity_state;
  view.identity_confidence = track.identity_confidence;
  return view;
}

BBox ExpandBox(const BBox& box, int width, int height) {
  const float box_width = box.x2 - box.x1;
  const float box_height = box.y2 - box.y1;
  BBox expanded{
      box.x1 - box_width * 0.125F,
      box.y1 - box_height * 0.2625F,
      box.x2 + box_width * 0.125F,
      box.y2 + box_height * 0.14F,
  };
  ClampBox(expanded, width, height, 1);
  return expanded;
}

void ProjectFaceGeometry(TrackEvent& track, const TrackEvent& previous,
                         std::int64_t capture_timestamp_ms,
                         const FaceConfig& config, int width, int height) {
  track.face = previous.face;
  track.face_track_id = previous.face_track_id;
  track.identity_id = previous.identity_id;
  track.identity_name = previous.identity_name;
  track.identity_state = previous.identity_state;
  track.identity_confidence = previous.identity_confidence;

  const auto& previous_body =
      previous.body.bbox.x2 > previous.body.bbox.x1 ? previous.body.bbox : previous.bbox;
  const float old_body_width = previous_body.x2 - previous_body.x1;
  const float old_body_height = previous_body.y2 - previous_body.y1;
  const float body_width = track.bbox.x2 - track.bbox.x1;
  const float body_height = track.bbox.y2 - track.bbox.y1;
  const auto old_face_box = previous.face.bbox;
  if (old_body_width > 1.0F && old_body_height > 1.0F &&
      body_width > 1.0F && body_height > 1.0F && IsFiniteBox(old_face_box)) {
    BBox projected{
        track.bbox.x1 + (old_face_box.x1 - previous_body.x1) / old_body_width * body_width,
        track.bbox.y1 + (old_face_box.y1 - previous_body.y1) / old_body_height * body_height,
        track.bbox.x1 + (old_face_box.x2 - previous_body.x1) / old_body_width * body_width,
        track.bbox.y1 + (old_face_box.y2 - previous_body.y1) / old_body_height * body_height,
    };
    if (ClampBox(projected, width, height, config.min_bbox_size_px)) {
      const float old_face_width = std::max(1.0F, old_face_box.x2 - old_face_box.x1);
      const float old_face_height = std::max(1.0F, old_face_box.y2 - old_face_box.y1);
      for (auto& landmark : track.face.landmarks_5pt) {
        landmark.first =
            projected.x1 + (landmark.first - old_face_box.x1) / old_face_width *
                               (projected.x2 - projected.x1);
        landmark.second =
            projected.y1 + (landmark.second - old_face_box.y1) / old_face_height *
                               (projected.y2 - projected.y1);
      }
      track.face.bbox = projected;
    }
  }

  track.face.face_bbox_observed = false;
  track.face.geometry_timestamp_ms = capture_timestamp_ms;
  track.face.geometry_age_ms =
      track.face.face_bbox_last_observed_ms > 0
          ? std::max<std::int64_t>(
                0, capture_timestamp_ms - track.face.face_bbox_last_observed_ms)
          : config.render_reacquire_hold_ms + 1;
  const auto continuity_ttl_ms =
      track.track_state == "tracked" ? config.tracked_asd_crop_ttl_ms
                                      : config.asd_crop_ttl_ms;
  if (track.face.face_bbox_last_observed_ms <= 0 ||
      track.face.geometry_age_ms > continuity_ttl_ms) {
    track.face.observation_state = "expired";
  } else if (track.face.geometry_age_ms <= config.hold_ms) {
    track.face.observation_state = "projected";
  } else {
    track.face.observation_state = "occluded";
  }

  track.render.bbox = ExpandBox(track.face.bbox, width, height);
  track.render.label =
      track.identity_name.empty() || track.identity_name == "unknown"
          ? "P" + std::to_string(track.person_track_id)
          : track.identity_name;
  const bool stable = track.track_state == "tracked" &&
                      track.face.geometry_age_ms <= config.render_green_hold_ms &&
                      (track.face.observation_state == "projected" ||
                       track.face.observation_state == "observed");
  const bool reacquiring =
      track.track_state != "lost" &&
      track.face.geometry_age_ms <= config.render_reacquire_hold_ms;
  track.render.state =
      stable ? "stable_tracking" : (reacquiring ? "reacquiring" : "occluded");
  track.render.color_state =
      stable ? "confirmed_good" : (reacquiring ? "tracking_ok" : "low_quality");
  track.render.show_glow = stable;
}

std::vector<float> ExtractAsdCrop(const cv::Mat& frame, const TrackEvent& track,
                                  const FaceConfig& config) {
  if (frame.empty()) {
    return {};
  }
  auto box = track.face.bbox;
  const float unclamped_width = box.x2 - box.x1;
  const float unclamped_height = box.y2 - box.y1;
  if (!IsFiniteBox(box) || unclamped_width < config.min_bbox_size_px ||
      unclamped_height < config.min_bbox_size_px ||
      track.face.geometry_age_ms > config.asd_crop_max_geometry_age_ms ||
      (track.face.observation_state != "observed" &&
       track.face.observation_state != "projected") ||
      track.face.landmarks_5pt.size() != 5U) {
    return {};
  }
  const auto visible_x1 = std::clamp(box.x1, 0.0F, static_cast<float>(frame.cols));
  const auto visible_y1 = std::clamp(box.y1, 0.0F, static_cast<float>(frame.rows));
  const auto visible_x2 = std::clamp(box.x2, 0.0F, static_cast<float>(frame.cols));
  const auto visible_y2 = std::clamp(box.y2, 0.0F, static_cast<float>(frame.rows));
  const auto visible_ratio =
      std::max(0.0F, visible_x2 - visible_x1) * std::max(0.0F, visible_y2 - visible_y1) /
      (unclamped_width * unclamped_height);
  if (visible_ratio < config.asd_crop_min_visible_ratio ||
      box.x1 < config.asd_crop_edge_margin_px ||
      box.y1 < config.asd_crop_edge_margin_px ||
      box.x2 > frame.cols - config.asd_crop_edge_margin_px ||
      box.y2 > frame.rows - config.asd_crop_edge_margin_px ||
      !ClampBox(box, frame.cols, frame.rows, config.min_bbox_size_px)) {
    return {};
  }
  const auto landmark_margin_x = (box.x2 - box.x1) * 0.20F;
  const auto landmark_margin_y = (box.y2 - box.y1) * 0.20F;
  const auto landmarks_valid = std::all_of(
      track.face.landmarks_5pt.begin(), track.face.landmarks_5pt.end(),
      [&box, landmark_margin_x, landmark_margin_y](const auto& landmark) {
        return std::isfinite(landmark.first) && std::isfinite(landmark.second) &&
               landmark.first >= box.x1 - landmark_margin_x &&
               landmark.first <= box.x2 + landmark_margin_x &&
               landmark.second >= box.y1 - landmark_margin_y &&
               landmark.second <= box.y2 + landmark_margin_y;
      });
  if (!landmarks_valid) {
    return {};
  }
  const float padding_x = (box.x2 - box.x1) * 0.15F;
  const float padding_y = (box.y2 - box.y1) * 0.15F;
  const int x1 = std::max(0, std::min(frame.cols - 1,
                                      static_cast<int>(std::round(box.x1 - padding_x))));
  const int y1 = std::max(0, std::min(frame.rows - 1,
                                      static_cast<int>(std::round(box.y1 - padding_y))));
  const int x2 = std::max(0, std::min(frame.cols,
                                      static_cast<int>(std::round(box.x2 + padding_x))));
  const int y2 = std::max(0, std::min(frame.rows,
                                      static_cast<int>(std::round(box.y2 + padding_y))));
  if (x2 - x1 < config.min_bbox_size_px || y2 - y1 < config.min_bbox_size_px) {
    return {};
  }
  cv::Mat gray;
  cv::cvtColor(frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)), gray, cv::COLOR_BGR2GRAY);
  cv::Mat resized;
  cv::resize(gray, resized, cv::Size(112, 112), 0.0, 0.0, cv::INTER_AREA);
  cv::Mat float_crop;
  resized.convertTo(float_crop, CV_32F);
  return std::vector<float>(
      reinterpret_cast<const float*>(float_crop.datastart),
      reinterpret_cast<const float*>(float_crop.dataend));
}

std::vector<float> CollectAudioWindow(
    const std::deque<AudioChunkEvent>& history,
    std::int64_t start_ms,
    std::int64_t end_ms,
    int sample_rate) {
  if (end_ms <= start_ms || sample_rate <= 0) {
    return {};
  }
  const auto target_size =
      static_cast<std::size_t>((end_ms - start_ms) * sample_rate / 1000);
  std::vector<float> samples(target_size, 0.0F);
  std::size_t copied = 0;
  for (const auto& chunk : history) {
    const auto chunk_end_ms = chunk.timestamp_ms + chunk.duration_ms;
    const auto overlap_start_ms = std::max(start_ms, chunk.timestamp_ms);
    const auto overlap_end_ms = std::min(end_ms, chunk_end_ms);
    if (overlap_end_ms <= overlap_start_ms) {
      continue;
    }
    const auto source_start =
        static_cast<std::size_t>((overlap_start_ms - chunk.timestamp_ms) * sample_rate / 1000);
    const auto target_start =
        static_cast<std::size_t>((overlap_start_ms - start_ms) * sample_rate / 1000);
    const auto length =
        static_cast<std::size_t>((overlap_end_ms - overlap_start_ms) * sample_rate / 1000);
    if (source_start >= chunk.data.size() || target_start >= samples.size()) {
      continue;
    }
    const auto safe_length =
        std::min({length, chunk.data.size() - source_start, samples.size() - target_start});
    std::copy_n(chunk.data.begin() + static_cast<std::ptrdiff_t>(source_start), safe_length,
                samples.begin() + static_cast<std::ptrdiff_t>(target_start));
    copied += safe_length;
  }
  return copied >= static_cast<std::size_t>(sample_rate * 0.35F) ? samples
                                                                 : std::vector<float>{};
}

void WriteLittleEndian16(std::ostream& output, std::uint16_t value) {
  output.put(static_cast<char>(value & 0xFF));
  output.put(static_cast<char>((value >> 8) & 0xFF));
}

void WriteLittleEndian32(std::ostream& output, std::uint32_t value) {
  output.put(static_cast<char>(value & 0xFF));
  output.put(static_cast<char>((value >> 8) & 0xFF));
  output.put(static_cast<char>((value >> 16) & 0xFF));
  output.put(static_cast<char>((value >> 24) & 0xFF));
}

void WriteWavHeader(std::ostream& output, int sample_rate,
                    std::uint32_t audio_bytes) {
  output.write("RIFF", 4);
  WriteLittleEndian32(output, 36U + audio_bytes);
  output.write("WAVEfmt ", 8);
  WriteLittleEndian32(output, 16);
  WriteLittleEndian16(output, 1);
  WriteLittleEndian16(output, 1);
  WriteLittleEndian32(output, static_cast<std::uint32_t>(sample_rate));
  WriteLittleEndian32(output, static_cast<std::uint32_t>(sample_rate * 2));
  WriteLittleEndian16(output, 2);
  WriteLittleEndian16(output, 16);
  output.write("data", 4);
  WriteLittleEndian32(output, audio_bytes);
}

}  // namespace

StreamingPipeline::StreamingPipeline(
    AppConfig config,
    std::shared_ptr<VisionBackend> vision,
    std::shared_ptr<VadBackend> vad,
    std::shared_ptr<AsrBackend> asr,
    std::shared_ptr<AsdBackend> asd,
    std::shared_ptr<FusionBackend> fusion,
    std::shared_ptr<DiarizationBackend> diarization)
    : config_(std::move(config)),
      vision_(std::move(vision)),
      vad_(std::move(vad)),
      asr_(std::move(asr)),
      asd_(std::move(asd)),
      fusion_(std::move(fusion)),
      diarization_(std::move(diarization)),
      frame_queue_(config_.runtime.frame_queue_size > 0 ? config_.runtime.frame_queue_size : 6),
      person_detector_queue_(1),
      person_detector_result_queue_(1),
      face_queue_(1),
      audio_queue_(static_cast<std::size_t>(
          std::max(1, config_.runtime.audio_queue_ms / std::max(1, config_.audio.chunk_ms)))),
      tracker_(
          static_cast<float>(config_.video.tracker_max_age_frames) /
              static_cast<float>(std::max(1, config_.video.fps)),
          static_cast<float>(config_.video.tracker_render_grace_ms) / 1000.0F,
          config_.video.tracker_high_confidence_threshold,
          config_.video.tracker_low_confidence_threshold,
          config_.video.tracker_iou_threshold,
          config_.video.tracker_min_hits,
          config_.video.tracker_center_distance_threshold) {
  const char* home = std::getenv("HOME");
  std::string home_dir = home ? std::string(home) : "";
  std::string face_model_dir = home_dir + "/.insightface/models/" + config_.face.model_pack;
  face_engine_ = std::make_unique<FaceEngine>(config_.face, face_model_dir, config_.runtime.use_cuda);
}

StreamingPipeline::~StreamingPipeline() {
  Stop();
}

bool StreamingPipeline::Start() {
  if (running_.exchange(true)) {
    return false;
  }
  metrics_started_ms_ = MonotonicNowMs();
  metrics_.session_replay_enabled = config_.diagnostics.session_replay_enabled;
  InitializeSessionReplay();

  video_thread_ = std::thread(&StreamingPipeline::VideoThreadLoop, this);
  person_detector_thread_ =
      std::thread(&StreamingPipeline::PersonDetectorThreadLoop, this);
  face_thread_ = std::thread(&StreamingPipeline::FaceThreadLoop, this);
  audio_thread_ = std::thread(&StreamingPipeline::AudioThreadLoop, this);
  fusion_thread_ = std::thread(&StreamingPipeline::FusionThreadLoop, this);

  return true;
}

void StreamingPipeline::Stop() {
  if (!running_.exchange(false)) {
    return;
  }

  frame_queue_.Close();
  person_detector_queue_.Close();
  person_detector_result_queue_.Close();
  face_queue_.Close();
  audio_queue_.Close();

  if (video_thread_.joinable()) {
    video_thread_.join();
  }
  if (person_detector_thread_.joinable()) {
    person_detector_thread_.join();
  }
  if (face_thread_.joinable()) {
    face_thread_.join();
  }
  if (audio_thread_.joinable()) {
    audio_thread_.join();
  }
  if (fusion_thread_.joinable()) {
    fusion_thread_.join();
  }
}

bool StreamingPipeline::PushFrame(FrameEvent frame) {
  if (!running_) return false;
  return frame_queue_.PushOrDrop(std::move(frame));
}

bool StreamingPipeline::PushAudio(AudioChunkEvent audio) {
  if (!running_) return false;
  return audio_queue_.Push(std::move(audio));
}

std::vector<FusionEvent> StreamingPipeline::GetLatestEvents() {
  std::lock_guard<std::mutex> lock(events_mu_);
  return {event_ring_.begin(), event_ring_.end()};
}

std::vector<FusionEvent> StreamingPipeline::ReadEventsAfter(
    std::uint64_t event_sequence) {
  std::lock_guard<std::mutex> lock(events_mu_);
  std::vector<FusionEvent> events;
  for (const auto& event : event_ring_) {
    if (event.event_sequence > event_sequence) {
      events.push_back(event);
    }
  }
  return events;
}

std::uint64_t StreamingPipeline::LatestEventSequence() {
  std::lock_guard<std::mutex> lock(events_mu_);
  return next_event_sequence_ > 1 ? next_event_sequence_ - 1 : 0;
}

std::vector<TrackEvent> StreamingPipeline::GetCurrentTracks() {
  std::lock_guard<std::mutex> lock(state_mu_);
  return current_tracks_;
}

PipelineMetrics StreamingPipeline::GetMetrics() {
  std::lock_guard<std::mutex> lock(metrics_mu_);
  auto metrics = metrics_;
  metrics.frame_queue_depth = frame_queue_.Size();
  metrics.frame_queue_dropped = frame_queue_.DroppedCount();
  metrics.face_queue_depth = face_queue_.Size();
  metrics.face_queue_dropped = face_queue_.DroppedCount();
  metrics.person_detector_queue_depth = person_detector_queue_.Size();
  metrics.person_detector_queue_dropped = person_detector_queue_.DroppedCount();
  metrics.audio_queue_depth = audio_queue_.Size();
  const auto elapsed_s =
      std::max(0.001, static_cast<double>(MonotonicNowMs() - metrics_started_ms_) / 1000.0);
  metrics.video_fps = static_cast<double>(metrics.video_frames_processed) / elapsed_s;
  metrics.person_detector_fps = static_cast<double>(metrics.person_detector_runs) / elapsed_s;
  metrics.face_pipeline_fps = static_cast<double>(metrics.face_pipeline_runs) / elapsed_s;
  metrics.asd_window_fps = static_cast<double>(metrics.asd_windows_scored) / elapsed_s;
  metrics.preview_encode_fps =
      static_cast<double>(metrics.preview_frames_encoded) / elapsed_s;
  return metrics;
}

void StreamingPipeline::ReportPreviewFrame(
    double encode_latency_ms, double capture_to_send_ms,
    std::uint64_t frames_dropped, std::uint64_t ws_dropped_frames,
    std::uint64_t pipe_frames_dropped, std::size_t pipe_backlog_bytes,
    const std::string& frame_age_source) {
  std::lock_guard<std::mutex> lock(metrics_mu_);
  ++metrics_.preview_frames_encoded;
  metrics_.preview_frames_dropped = frames_dropped;
  metrics_.preview_ws_dropped_frames = ws_dropped_frames;
  metrics_.pipe_frames_dropped = pipe_frames_dropped;
  metrics_.pipe_backlog_bytes = pipe_backlog_bytes;
  metrics_.preview_frame_age_source = frame_age_source;
  RecordLatency(preview_encode_latencies_ms_, encode_latency_ms);
  RecordLatency(preview_capture_to_send_ms_, capture_to_send_ms);
  metrics_.preview_encode_latency_p95_ms =
      Percentile(preview_encode_latencies_ms_, 0.95);
  metrics_.preview_capture_to_send_p95_ms =
      Percentile(preview_capture_to_send_ms_, 0.95);
}

std::string StreamingPipeline::GetLatestError() {
  std::lock_guard<std::mutex> lock(error_mu_);
  return latest_error_;
}

void StreamingPipeline::RecordLatency(std::deque<double>& values, double latency_ms) {
  values.push_back(latency_ms);
  while (values.size() > 256U) {
    values.pop_front();
  }
}

void StreamingPipeline::EnqueueEvents(std::vector<FusionEvent> events) {
  if (events.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(events_mu_);
  for (auto& event : events) {
    event.event_sequence = next_event_sequence_++;
    event_ring_.push_back(std::move(event));
    while (event_ring_.size() > config_.runtime.event_ring_size) {
      event_ring_.pop_front();
    }
  }
}

void StreamingPipeline::RecordSessionLine(const std::string& json_line) {
  if (config_.runtime.session_log_path.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(recorder_mu_);
  const std::filesystem::path path(config_.runtime.session_log_path);
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::app);
  if (!output) {
    SetLatestError("session recorder: cannot append " + path.string());
    return;
  }
  output << json_line << '\n';
}

void StreamingPipeline::InitializeSessionReplay() {
  if (!config_.diagnostics.session_replay_enabled) {
    return;
  }
  std::lock_guard<std::mutex> lock(recorder_mu_);
  const std::filesystem::path replay_dir(config_.diagnostics.session_replay_dir);
  std::filesystem::create_directories(replay_dir);
  const auto cutoff =
      std::filesystem::file_time_type::clock::now() -
      std::chrono::minutes(config_.diagnostics.session_replay_ttl_minutes);
  for (const auto& entry : std::filesystem::directory_iterator(replay_dir)) {
    std::error_code error;
    const auto modified = std::filesystem::last_write_time(entry.path(), error);
    if (!error && modified < cutoff) {
      std::filesystem::remove_all(entry.path(), error);
    }
  }
  std::filesystem::remove(replay_dir / "session_audio.wav");
  std::filesystem::remove(replay_dir / "session_timeline.jsonl");
  replay_audio_bytes_ = 0;
  replay_last_crop_ms_.clear();
}

void StreamingPipeline::RecordReplayAudio(const AudioChunkEvent& audio) {
  if (!config_.diagnostics.session_replay_enabled || audio.data.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(recorder_mu_);
  const auto replay_dir = std::filesystem::path(config_.diagnostics.session_replay_dir);
  std::filesystem::create_directories(replay_dir);
  const auto wav_path = replay_dir / "session_audio.wav";
  std::fstream wav(wav_path, std::ios::binary | std::ios::in | std::ios::out);
  if (!wav) {
    std::ofstream create(wav_path, std::ios::binary);
    WriteWavHeader(create, audio.sample_rate, 0);
    create.close();
    wav.open(wav_path, std::ios::binary | std::ios::in | std::ios::out);
  }
  wav.seekp(0, std::ios::end);
  for (const float sample : audio.data) {
    const auto pcm = static_cast<std::int16_t>(std::clamp(
        std::lround(std::clamp(sample, -1.0F, 1.0F) * 32768.0F),
        -32768L, 32767L));
    WriteLittleEndian16(wav, static_cast<std::uint16_t>(pcm));
  }
  replay_audio_bytes_ += audio.data.size() * sizeof(std::int16_t);
  wav.seekp(0, std::ios::beg);
  WriteWavHeader(wav, audio.sample_rate,
                 static_cast<std::uint32_t>(replay_audio_bytes_));
  std::ofstream timeline(replay_dir / "session_timeline.jsonl", std::ios::app);
  timeline << "{\"type\":\"audio\",\"timestamp_ms\":" << audio.timestamp_ms
           << ",\"duration_ms\":" << audio.duration_ms
           << ",\"sample_count\":" << audio.data.size()
           << ",\"clock_drift_ms\":" << audio.clock_drift_ms << "}\n";
}

void StreamingPipeline::RecordReplayCrop(
    std::int64_t timestamp_ms, int person_track_id,
    const std::vector<float>& grayscale_112) {
  if (!config_.diagnostics.session_replay_enabled ||
      grayscale_112.size() != 112U * 112U ||
      timestamp_ms - replay_last_crop_ms_[person_track_id] < 100) {
    return;
  }
  std::lock_guard<std::mutex> lock(recorder_mu_);
  replay_last_crop_ms_[person_track_id] = timestamp_ms;
  const auto replay_dir = std::filesystem::path(config_.diagnostics.session_replay_dir);
  std::filesystem::create_directories(replay_dir);
  const auto filename = "crop_" + std::to_string(timestamp_ms) + "_P" +
                        std::to_string(person_track_id) + ".pgm";
  WritePgm(replay_dir / filename, grayscale_112);
  std::ofstream timeline(replay_dir / "session_timeline.jsonl", std::ios::app);
  timeline << "{\"type\":\"crop\",\"timestamp_ms\":" << timestamp_ms
           << ",\"person_track_id\":" << person_track_id
           << ",\"path\":\"" << filename << "\"}\n";
}

void StreamingPipeline::RecordReplayWindow(
    const AsdInputWindow& window,
    const std::vector<ActiveSpeakerScore>& scores) {
  if (!config_.diagnostics.session_replay_enabled) {
    return;
  }
  const auto mfcc = ExtractLrAsdMfcc(window.audio_samples, window.sample_rate);
  std::lock_guard<std::mutex> lock(recorder_mu_);
  const auto replay_dir = std::filesystem::path(config_.diagnostics.session_replay_dir);
  std::filesystem::create_directories(replay_dir);
  std::ofstream timeline(replay_dir / "session_timeline.jsonl", std::ios::app);
  timeline << "{\"type\":\"asd_manifest\",\"start_ms\":" << window.start_ms
           << ",\"end_ms\":" << window.end_ms
           << ",\"audio_hash\":\"" << std::hex << HashFloats(window.audio_samples)
           << "\",\"mfcc_hash\":\"" << HashFloats(mfcc) << std::dec
           << "\",\"tracks\":[";
  bool first_track = true;
  for (const auto& track : window.tracks) {
    if (!first_track) {
      timeline << ",";
    }
    first_track = false;
    const auto visual = window.visual_samples.find(track.person_track_id);
    const auto timestamps =
        visual == window.visual_samples.end()
            ? std::vector<std::int64_t>{}
            : SelectVisualTimestamps(visual->second, window.start_ms, window.end_ms);
    const auto score = std::find_if(scores.begin(), scores.end(), [&track](const auto& item) {
      return item.person_track_id == track.person_track_id;
    });
    timeline << "{\"person_track_id\":" << track.person_track_id
             << ",\"raw_p_active\":"
             << (score == scores.end() ? 0.0F : score->raw_p_active)
             << ",\"smoothed_p_active\":"
             << (score == scores.end() ? 0.0F : score->p_active)
             << ",\"crop_timestamps_ms\":[";
    for (std::size_t index = 0; index < timestamps.size(); ++index) {
      if (index > 0) {
        timeline << ",";
      }
      timeline << timestamps[index];
    }
    timeline << "],\"crop_paths\":[";
    for (std::size_t index = 0; index < timestamps.size(); ++index) {
      if (index > 0) {
        timeline << ",";
      }
      const auto filename =
          "window_crop_" + std::to_string(timestamps[index]) + "_P" +
          std::to_string(track.person_track_id) + ".pgm";
      const auto path = replay_dir / filename;
      if (!std::filesystem::exists(path) && visual != window.visual_samples.end()) {
        const auto sample = std::min_element(
            visual->second.begin(), visual->second.end(),
            [timestamp = timestamps[index]](const auto& left, const auto& right) {
              return std::llabs(left.timestamp_ms - timestamp) <
                     std::llabs(right.timestamp_ms - timestamp);
            });
        if (sample != visual->second.end()) {
          WritePgm(path, sample->grayscale_112);
        }
      }
      timeline << "\"" << filename << "\"";
    }
    timeline << "]}";
  }
  timeline << "]}\n";
}

void StreamingPipeline::SetLatestError(const std::string& error) {
  std::lock_guard<std::mutex> lock(error_mu_);
  latest_error_ = error;
}


void StreamingPipeline::VideoThreadLoop() {
  int frame_count = 0;
  std::int64_t last_visual_snapshot_ms = 0;
  while (running_) {
    auto frame_opt = frame_queue_.Pop();
    if (!frame_opt) {
      break;
    }

    try {
      std::vector<TrackEvent> tracks;
      int skip_n = config_.video.person_detector_detect_every_n_frames;
      if (skip_n <= 0) skip_n = 1;
      if (frame_count % skip_n == 0) {
        person_detector_queue_.PushOrDrop(*frame_opt);
      }
      auto detection_result = person_detector_result_queue_.TryPop();
      while (auto newer_result = person_detector_result_queue_.TryPop()) {
        detection_result = std::move(newer_result);
      }
      const bool has_fresh_detection =
          detection_result &&
          frame_opt->timestamp_ms >= detection_result->capture_timestamp_ms &&
          frame_opt->timestamp_ms - detection_result->capture_timestamp_ms <=
              config_.video.max_detection_result_age_ms;
      if (has_fresh_detection) {
        {
          std::lock_guard<std::mutex> lock(tracker_mu_);
          tracks = tracker_.Update(detection_result->detections, frame_opt->width,
                                   frame_opt->height);
        }
        if (!tracks.empty()) {
          static int print_count = 0;
          if (print_count < 10) {
            std::cout << "[Pipeline] Tracker updated: " << tracks.size() << " visible tracks\n";
            for (const auto& t : tracks) {
              std::cout << "  - Track ID " << t.person_track_id 
                        << ", Box=[" << t.bbox.x1 << ", " << t.bbox.y1 << ", " << t.bbox.x2 << ", " << t.bbox.y2 << "]\n";
            }
            print_count++;
          }
        }
      } else {
        std::lock_guard<std::mutex> lock(tracker_mu_);
        tracks = tracker_.PredictOnly(frame_opt->width, frame_opt->height);
      }

      // Decode frame image for FaceEngine
      cv::Mat frame_mat;
      if (frame_opt->data.size() > 4 && frame_opt->data[0] == 0xFF && frame_opt->data[1] == 0xD8) {
        frame_mat = cv::imdecode(frame_opt->data, cv::IMREAD_COLOR);
      } else if (frame_opt->width > 0 && frame_opt->height > 0 && frame_opt->data.size() == static_cast<size_t>(frame_opt->width * frame_opt->height * 3)) {
        frame_mat = cv::Mat(frame_opt->height, frame_opt->width, CV_8UC3, const_cast<uint8_t*>(frame_opt->data.data())).clone();
      }

      const auto visual_timestamp_ms =
          frame_opt->timestamp_ms +
          static_cast<std::int64_t>(std::llround(config_.sync.video_time_offset_ms));
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        for (auto& track : tracks) {
          track.track_snapshot_sequence = next_track_snapshot_sequence_++;
          track.snapshot_timestamp_ms = frame_opt->timestamp_ms;
          track.body.bbox = track.bbox;
          track.body.quality = track.quality;
          track.body.source = "yolo26_person";
          const auto previous = std::find_if(
              current_tracks_.begin(), current_tracks_.end(), [&track](const auto& item) {
                return item.person_track_id == track.person_track_id;
              });
          if (previous != current_tracks_.end()) {
            ProjectFaceGeometry(track, *previous, frame_opt->timestamp_ms, config_.face,
                                frame_opt->width, frame_opt->height);
          }
        }
        current_tracks_ = tracks;
        const auto oldest_visual_ms =
            visual_timestamp_ms - std::max(4000, config_.asd.window_ms + 1000);
        if (!frame_mat.empty()) {
          for (const auto& track : current_tracks_) {
            const auto crop_ttl_ms =
                track.track_state == "tracked"
                    ? config_.face.tracked_asd_crop_ttl_ms
                    : config_.face.asd_crop_ttl_ms;
            const bool crop_allowed =
                track.face.face_bbox_last_observed_ms > 0 &&
                frame_opt->timestamp_ms - track.face.face_bbox_last_observed_ms <=
                    crop_ttl_ms;
            if (!crop_allowed) {
              continue;
            }
            auto crop =
                ExtractAsdCrop(frame_mat, track, config_.face);
            if (!crop.empty()) {
              RecordReplayCrop(visual_timestamp_ms, track.person_track_id, crop);
              asd_visual_history_[track.person_track_id].push_back(
                  AsdVisualSample{visual_timestamp_ms, std::move(crop)});
            }
          }
        }
        for (auto& [_, history] : asd_visual_history_) {
          while (!history.empty() && history.front().timestamp_ms < oldest_visual_ms) {
            history.pop_front();
          }
        }
      }

      if (!frame_mat.empty() && face_engine_ &&
          frame_count % std::max(1, config_.face.run_every_n_frames) == 0) {
        face_queue_.PushOrDrop(
            FaceFrameJob{std::move(frame_mat), tracks, frame_count, frame_opt->timestamp_ms});
      }
      frame_count++;
      {
        std::lock_guard<std::mutex> lock(metrics_mu_);
        ++metrics_.video_frames_processed;
        metrics_.latest_visual_watermark_ms = visual_timestamp_ms;
        metrics_.visual_watermark_lag_ms =
            metrics_.latest_audio_watermark_ms > 0
                ? metrics_.latest_audio_watermark_ms - visual_timestamp_ms
                : 0;
        metrics_.watermark_lag_ms = metrics_.visual_watermark_lag_ms;
        metrics_.sync_healthy =
            metrics_.latest_audio_watermark_ms == 0 ||
            std::llabs(metrics_.visual_watermark_lag_ms) <=
                config_.sync.max_watermark_lag_ms;
      }
      if (frame_opt->timestamp_ms - last_visual_snapshot_ms >= 80) {
        FusionEvent snapshot;
        snapshot.event_type = "track_update";
        snapshot.position = "offscreen";
        snapshot.tentative = true;
        snapshot.decision_reason = "visual_track_update";
        {
          std::lock_guard<std::mutex> lock(state_mu_);
          for (const auto& track : current_tracks_) {
            if (!HasRenderableFace(track)) {
              continue;
            }
            float p_active = 0.0F;
            for (const auto& score : current_asd_scores_) {
              if (score.person_track_id == track.person_track_id) {
                p_active = std::max(p_active, score.p_active);
              }
            }
            snapshot.tracks.push_back(MakeTrackView(track, p_active));
          }
        }
        EnqueueEvents({std::move(snapshot)});
        last_visual_snapshot_ms = frame_opt->timestamp_ms;
      }
    } catch (const std::exception& e) {
      SetLatestError(std::string("video pipeline: ") + e.what());
      std::cerr << "Error in VideoThreadLoop: " << e.what() << "\n";
    }
  }
}

void StreamingPipeline::PersonDetectorThreadLoop() {
  while (running_) {
    auto frame = person_detector_queue_.Pop();
    if (!frame) {
      break;
    }
    try {
      const auto started_at = std::chrono::steady_clock::now();
      const auto raw_detections = vision_->AcceptFrame(*frame);
      PersonDetectorResult result;
      result.capture_timestamp_ms = frame->timestamp_ms;
      result.width = frame->width;
      result.height = frame->height;
      result.detections.reserve(raw_detections.size());
      for (const auto& detection : raw_detections) {
        result.detections.push_back(
            PersonDetection{detection.bbox, detection.confidence});
      }
      person_detector_result_queue_.PushOrDrop(std::move(result));
      const auto latency_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started_at)
              .count();
      std::lock_guard<std::mutex> lock(metrics_mu_);
      ++metrics_.person_detector_runs;
      RecordLatency(person_detector_latencies_ms_, latency_ms);
      metrics_.person_detector_latency_p50_ms =
          Percentile(person_detector_latencies_ms_, 0.50);
      metrics_.person_detector_latency_p95_ms =
          Percentile(person_detector_latencies_ms_, 0.95);
    } catch (const std::exception& error) {
      SetLatestError(std::string("person detector: ") + error.what());
      std::cerr << "Error in PersonDetectorThreadLoop: " << error.what() << "\n";
    }
  }
}

void StreamingPipeline::FaceThreadLoop() {
  while (running_) {
    auto job = face_queue_.Pop();
    if (!job) {
      break;
    }
    try {
      const auto face_started_at = std::chrono::steady_clock::now();
      auto tracks = face_engine_->UpdateTracks(
          job->frame, job->tracks, job->frame_count, job->capture_timestamp_ms);
      {
        std::lock_guard<std::mutex> lock(tracker_mu_);
        tracker_.ApplyIdentityHints(tracks);
      }
      const double face_pipeline_latency_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - face_started_at)
              .count();
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (current_tracks_.empty()) {
          current_tracks_ = tracks;
        } else {
          for (auto& current : current_tracks_) {
            const auto enriched = std::find_if(
                tracks.begin(), tracks.end(), [&current](const auto& item) {
                  return item.person_track_id == current.person_track_id;
                });
            if (enriched == tracks.end()) {
              continue;
            }
            if (enriched->face.face_bbox_last_observed_ms >
                current.face.face_bbox_last_observed_ms) {
              ProjectFaceGeometry(
                  current, *enriched, current.snapshot_timestamp_ms, config_.face,
                  job->frame.cols, job->frame.rows);
            } else if (enriched->face.face_bbox_last_observed_ms <
                       current.face.face_bbox_last_observed_ms) {
              std::lock_guard<std::mutex> metrics_lock(metrics_mu_);
              ++metrics_.stale_geometry_rejected;
            }
            current.face_track_id = enriched->face_track_id;
            current.identity_id = enriched->identity_id;
            current.identity_name = enriched->identity_name;
            current.identity_state = enriched->identity_state;
            current.identity_confidence = enriched->identity_confidence;
          }
        }
      }
      std::size_t directly_observed = 0;
      for (const auto& track : tracks) {
        if (track.face.observation_state == "observed") {
          ++directly_observed;
        }
      }
      {
        std::lock_guard<std::mutex> lock(metrics_mu_);
        ++metrics_.face_pipeline_runs;
        metrics_.direct_face_observation_ratio =
            tracks.empty()
                ? 0.0F
                : static_cast<float>(directly_observed) /
                      static_cast<float>(tracks.size());
        RecordLatency(face_pipeline_latencies_ms_, face_pipeline_latency_ms);
        metrics_.face_pipeline_latency_p50_ms =
            Percentile(face_pipeline_latencies_ms_, 0.50);
        metrics_.face_pipeline_latency_p95_ms =
            Percentile(face_pipeline_latencies_ms_, 0.95);
      }
    } catch (const std::exception& error) {
      SetLatestError(std::string("face pipeline: ") + error.what());
      std::cerr << "Error in FaceThreadLoop: " << error.what() << "\n";
    }
  }
}

void StreamingPipeline::AudioThreadLoop() {
  bool was_speech_active = false;
  int64_t current_utterance_start_ms = 0;
  int64_t last_partial_emit_time_ms = 0;
  std::string last_partial_text = "";
  int64_t accumulated_samples = 0;

  while (running_) {
    auto audio_opt = audio_queue_.Pop();
    if (!audio_opt) {
      break;
    }

    try {
      RecordReplayAudio(*audio_opt);
      asr_->PushAudio(*audio_opt);

      // Keep sliding window of latest chunks for preroll pre-speech context
      preroll_buffer_.push_back(audio_opt->data);
      const auto preroll_chunks = static_cast<std::size_t>(
          std::max(1, (config_.vad.preroll_ms + config_.audio.chunk_ms - 1) /
                          config_.audio.chunk_ms));
      if (preroll_buffer_.size() > preroll_chunks) {
        preroll_buffer_.pop_front();
      }

      auto vad_segments = vad_->AcceptAudio(*audio_opt);
      bool is_speech_active = vad_->IsSpeechActive();
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (audio_opt->clock_reanchored) {
          asd_audio_history_.clear();
          asd_visual_history_.clear();
          asd_timeline_.clear();
          current_asd_scores_.clear();
        }
        asd_audio_history_.push_back(*audio_opt);
        latest_audio_end_ms_ = audio_opt->timestamp_ms + audio_opt->duration_ms;
        speech_active_ = is_speech_active;
        if (is_speech_active) {
          last_speech_active_ms_ = latest_audio_end_ms_;
        }
        const auto oldest_audio_ms =
            latest_audio_end_ms_ - std::max(4000, config_.asd.window_ms + 1000);
        while (!asd_audio_history_.empty() &&
               asd_audio_history_.front().timestamp_ms +
                       asd_audio_history_.front().duration_ms <
                   oldest_audio_ms) {
          asd_audio_history_.pop_front();
        }
      }
      {
        std::lock_guard<std::mutex> lock(metrics_mu_);
        if (audio_opt->clock_reanchored) {
          ++metrics_.sync_reanchor_count;
        }
        metrics_.drift_ms_per_min = audio_opt->clock_drift_ms_per_min;
        metrics_.latest_audio_watermark_ms = audio_opt->timestamp_ms + audio_opt->duration_ms;
        metrics_.visual_watermark_lag_ms =
            metrics_.latest_visual_watermark_ms > 0
                ? metrics_.latest_audio_watermark_ms - metrics_.latest_visual_watermark_ms
                : 0;
        metrics_.watermark_lag_ms = metrics_.visual_watermark_lag_ms;
        metrics_.sync_healthy =
            metrics_.latest_visual_watermark_ms == 0 ||
            std::llabs(metrics_.visual_watermark_lag_ms) <=
                config_.sync.max_watermark_lag_ms;
      }

      // 1. Detect speech start and trigger ASR streaming start
      if (is_speech_active && !was_speech_active) {
        asr_->BeginStreaming();
        current_utterance_start_ms =
            std::max<std::int64_t>(0, audio_opt->timestamp_ms - config_.vad.preroll_ms);
        last_partial_emit_time_ms = 0;
        last_partial_text = "";
        accumulated_samples = 0;

        // Start collecting speech samples with buffered preroll audio
        current_speech_samples_.clear();
        for (auto iterator = preroll_buffer_.begin(); iterator != preroll_buffer_.end();
             ++iterator) {
          if (std::next(iterator) == preroll_buffer_.end()) {
            break;
          }
          current_speech_samples_.insert(
              current_speech_samples_.end(), iterator->begin(), iterator->end());
        }
      }

      // 2. Transcribe stream chunks while speaking (ASR Partial)
      if (is_speech_active) {
        // Collect current speech samples
        current_speech_samples_.insert(current_speech_samples_.end(), audio_opt->data.begin(), audio_opt->data.end());

        accumulated_samples += audio_opt->data.size();
        auto partial_res = asr_->StreamAudio(audio_opt->data, audio_opt->sample_rate);
        std::string text = partial_res.text;

        int min_audio_ms = config_.asr.partial_min_audio_ms > 0 ? config_.asr.partial_min_audio_ms : 480;
        int interval_ms = config_.asr.partial_interval_ms > 0 ? config_.asr.partial_interval_ms : 400;

        int64_t audio_duration_ms = accumulated_samples * 1000 / audio_opt->sample_rate;
        int64_t now_ms = MonotonicNowMs();

        if (!text.empty() && text != last_partial_text && audio_duration_ms >= min_audio_ms &&
            (now_ms - last_partial_emit_time_ms >= interval_ms)) {
          last_partial_emit_time_ms = now_ms;
          const std::string previous_text = last_partial_text;
          last_partial_text = text;
          std::cout << "[AudioPipeline] ASR partial: \"" << text << "\" (" << audio_duration_ms << "ms)\n";

          UtteranceEvent ev;
          ev.utterance_id = "utt_partial_" + std::to_string(current_utterance_start_ms);
          ev.start_ms = current_utterance_start_ms;
          ev.end_ms = audio_opt->timestamp_ms + audio_opt->duration_ms;
          ev.text = text;
          ev.text_delta =
              text.rfind(previous_text, 0) == 0 ? text.substr(previous_text.size()) : text;
          ev.final = false;
          ev.confidence = partial_res.confidence;
          ev.stability = partial_res.stability;
          ev.tokens = partial_res.tokens;
          ev.token_timestamps_s = partial_res.token_timestamps_s;

          {
            std::lock_guard<std::mutex> lock(state_mu_);
            // Replace any existing partial utterance with the same ID, or append
            auto it = std::find_if(current_utterances_.begin(), current_utterances_.end(),
                                   [&ev](const UtteranceEvent& u) { return u.utterance_id == ev.utterance_id; });
            if (it != current_utterances_.end()) {
              *it = ev;
            } else {
              current_utterances_.push_back(ev);
            }
          }
        }
      }

      // 3. Detect speech end and close streaming recognizer
      if (!is_speech_active && was_speech_active) {
        asr_->EndStreaming();
      }

      was_speech_active = is_speech_active;

      // 4. Handle final VAD segments (on silent pauses)
      if (!vad_segments.empty()) {
        std::cout << "[AudioPipeline] VAD detected " << vad_segments.size() << " speech segment(s):\n";
        std::vector<UtteranceEvent> utterances;
        for (const auto& seg : vad_segments) {
          std::cout << "  - Segment: " << seg.start_ms << " -> " << seg.end_ms << " ms\n";
          // Pass the dynamically collected speech samples directly to avoid timestamp offset drifts
          auto utts = asr_->AcceptSpeechSegment(seg, current_speech_samples_, diarization_.get());
          for (const auto& utt : utts) {
            std::cout << "    * ASR transcribed: \"" << utt.text << "\"\n";
          }
          utterances.insert(utterances.end(), utts.begin(), utts.end());
        }

        // Clean collected speech samples after turn transcription is dispatched
        current_speech_samples_.clear();

        {
          std::lock_guard<std::mutex> lock(state_mu_);
          current_vad_segments_.insert(current_vad_segments_.end(), vad_segments.begin(), vad_segments.end());

          // Erase the old temporary partial utterance to avoid duplicate/stale display
          std::string partial_id = "utt_partial_" + std::to_string(current_utterance_start_ms);
          current_utterances_.erase(
              std::remove_if(current_utterances_.begin(), current_utterances_.end(),
                             [&partial_id](const UtteranceEvent& u) { return u.utterance_id == partial_id; }),
              current_utterances_.end());

          current_utterances_.insert(current_utterances_.end(), utterances.begin(), utterances.end());
        }
      }
    } catch (const std::exception& e) {
      SetLatestError(std::string("audio pipeline: ") + e.what());
      std::cerr << "Error in AudioThreadLoop: " << e.what() << "\n";
    }
  }
}

void StreamingPipeline::FusionThreadLoop() {
  const int interval_ms = config_.asd.hop_ms > 0 ? config_.asd.hop_ms : 320;
  
  while (running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));

    FusionInput input;
    AsdInputWindow window;
    std::int64_t last_speech_active_ms = 0;
    std::size_t person_track_count = 0;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      input.tracks = current_tracks_;
      person_track_count = current_tracks_.size();
      input.tracks.erase(
          std::remove_if(input.tracks.begin(), input.tracks.end(),
                         [](const auto& track) { return !HasRenderableFace(track); }),
          input.tracks.end());
      input.vad_segments = std::move(current_vad_segments_);
      input.utterances = std::move(current_utterances_);
      current_vad_segments_.clear();
      current_utterances_.clear();
      window.end_ms = latest_audio_end_ms_;
      window.start_ms = std::max<std::int64_t>(0, window.end_ms - config_.asd.window_ms);
      window.sample_rate = config_.audio.sample_rate;
      window.speech_active = speech_active_;
      last_speech_active_ms = last_speech_active_ms_;
      window.audio_samples =
          CollectAudioWindow(asd_audio_history_, window.start_ms, window.end_ms, window.sample_rate);
      for (const auto& [track_id, history] : asd_visual_history_) {
        auto& samples = window.visual_samples[track_id];
        for (const auto& sample : history) {
          if (sample.timestamp_ms >= window.start_ms - 250 &&
              sample.timestamp_ms <= window.end_ms + 250) {
            samples.push_back(sample);
          }
        }
      }
    }

    if (input.tracks.empty() && input.vad_segments.empty() && input.utterances.empty()) {
      continue;
    }

    try {
      const bool pending_final = std::any_of(
          input.utterances.begin(), input.utterances.end(),
          [](const auto& utterance) { return utterance.final; });
      const bool vad_gate_open =
          window.speech_active || pending_final ||
          (last_speech_active_ms > 0 &&
           window.end_ms - last_speech_active_ms <= config_.asd.vad_hangover_ms);

      struct Candidate {
        TrackEvent track;
        int crop_count = 0;
        int max_crop_gap_ms = 0;
      };
      std::vector<Candidate> candidates;
      for (const auto& track : input.tracks) {
        const auto visual = window.visual_samples.find(track.person_track_id);
        if (track.track_state != "tracked" ||
            track.face.face_bbox_last_observed_ms <= 0 ||
            track.face.observation_state == "expired" ||
            visual == window.visual_samples.end()) {
          continue;
        }
        const auto crop_count = static_cast<int>(visual->second.size());
        const auto max_crop_gap_ms = MaxCropGapMs(visual->second);
        if (crop_count < config_.asd.min_crop_count ||
            max_crop_gap_ms > config_.asd.max_crop_gap_ms) {
          continue;
        }
        candidates.push_back(Candidate{track, crop_count, max_crop_gap_ms});
      }
      std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        if (left.crop_count != right.crop_count) {
          return left.crop_count > right.crop_count;
        }
        return left.track.face.face_bbox_last_observed_ms >
               right.track.face.face_bbox_last_observed_ms;
      });
      if (candidates.size() > static_cast<std::size_t>(config_.asd.max_candidate_tracks)) {
        candidates.resize(static_cast<std::size_t>(config_.asd.max_candidate_tracks));
      }
      window.tracks.clear();
      for (const auto& candidate : candidates) {
        window.tracks.push_back(candidate.track);
      }
      window.vad_segments = input.vad_segments;

      bool sync_healthy = true;
      {
        std::lock_guard<std::mutex> lock(metrics_mu_);
        sync_healthy = metrics_.sync_healthy;
      }
      const auto asd_started_at = std::chrono::steady_clock::now();
      const bool should_score = sync_healthy && vad_gate_open && !window.tracks.empty();
      auto scores = should_score ? asd_->ScoreWindow(window)
                                 : std::vector<ActiveSpeakerScore>{};
      if (should_score) {
        RecordReplayWindow(window, scores);
      }
      if (!sync_healthy) {
        input.asd_failure_reason = "sync_unhealthy";
      } else if (!vad_gate_open) {
        input.asd_failure_reason = "vad_gate_closed";
      } else if (window.tracks.empty()) {
        input.asd_failure_reason = "no_face_backed_asd_candidate";
      }
      const double asd_latency_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - asd_started_at)
              .count();
      std::int64_t visual_watermark_lag_ms = 0;
      {
        std::lock_guard<std::mutex> lock(metrics_mu_);
        metrics_.person_track_count = person_track_count;
        metrics_.rendered_face_track_count = input.tracks.size();
        metrics_.asd_candidate_count = window.tracks.size();
        metrics_.no_face_track_count =
            metrics_.person_track_count >= metrics_.rendered_face_track_count
                ? metrics_.person_track_count - metrics_.rendered_face_track_count
                : 0;
        if (should_score) {
          RecordLatency(asd_latencies_ms_, asd_latency_ms);
          ++metrics_.asd_windows_scored;
        } else if (!vad_gate_open) {
          ++metrics_.asd_vad_skipped_windows;
        }
        if (!sync_healthy) {
          ++metrics_.asd_gated_windows;
        }
        metrics_.asd_latency_p50_ms = Percentile(asd_latencies_ms_, 0.50);
        metrics_.asd_latency_p95_ms = Percentile(asd_latencies_ms_, 0.95);
        visual_watermark_lag_ms = metrics_.visual_watermark_lag_ms;
        if (window.speech_active) {
          ++asd_windows_total_;
          if (!scores.empty()) {
            ++asd_windows_usable_;
          }
          metrics_.asd_usable_window_ratio =
              asd_windows_total_ == 0
                  ? 0.0F
                  : static_cast<float>(asd_windows_usable_) /
                        static_cast<float>(asd_windows_total_);
        }
      }

      std::ostringstream asd_record;
      asd_record << "{\"type\":\"asd_window\",\"start_ms\":" << window.start_ms
                 << ",\"end_ms\":" << window.end_ms
                 << ",\"speech_active\":" << (window.speech_active ? "true" : "false")
                 << ",\"audio_samples\":" << window.audio_samples.size()
                 << ",\"visual_watermark_lag_ms\":" << visual_watermark_lag_ms
                 << ",\"tracks\":[";
      bool first_track = true;
      for (const auto& track : input.tracks) {
        const auto visual = window.visual_samples.find(track.person_track_id);
        const int crop_count =
            visual == window.visual_samples.end() ? 0 : static_cast<int>(visual->second.size());
        const int max_crop_gap_ms =
            visual == window.visual_samples.end() ? 0 : MaxCropGapMs(visual->second);
        auto scored = std::find_if(scores.begin(), scores.end(), [&track](const auto& score) {
          return score.person_track_id == track.person_track_id;
        });
        std::string failure_reason;
        if (!sync_healthy) {
          failure_reason = "sync_unhealthy";
        } else if (!vad_gate_open) {
          failure_reason = "vad_gate_closed";
        } else if (window.audio_samples.empty()) {
          failure_reason = "missing_audio_window";
        } else if (crop_count == 0) {
          failure_reason = "missing_visual_crops";
        } else if (crop_count < config_.asd.min_crop_count ||
                   max_crop_gap_ms > config_.asd.max_crop_gap_ms) {
          failure_reason = "insufficient_visual_coverage";
        } else if (scored == scores.end()) {
          failure_reason = "not_selected_asd_candidate";
        }
        if (scored != scores.end()) {
          scored->crop_count = crop_count;
          scored->max_crop_gap_ms = max_crop_gap_ms;
          scored->visual_watermark_lag_ms = visual_watermark_lag_ms;
        }
        if (!first_track) {
          asd_record << ",";
        }
        first_track = false;
        asd_record << "{\"person_track_id\":" << track.person_track_id
                   << ",\"crop_count\":" << crop_count
                   << ",\"max_crop_gap_ms\":" << max_crop_gap_ms
                   << ",\"raw_p_active\":"
                   << (scored == scores.end() ? 0.0F : scored->raw_p_active)
                   << ",\"smoothed_p_active\":"
                   << (scored == scores.end() ? 0.0F : scored->p_active)
                   << ",\"failure_reason\":\"" << EscapeJsonString(failure_reason) << "\"}";
      }
      asd_record << "]}";
      RecordSessionLine(asd_record.str());

      input.active_speaker_scores = scores;
      if (should_score) {
        asd_timeline_.push_back(AsdTimelinePoint{window.end_ms, scores});
      }
      const auto oldest_timeline_ms =
          window.end_ms - static_cast<std::int64_t>(config_.timeline.history_s * 1000.0F);
      while (!asd_timeline_.empty() && asd_timeline_.front().timestamp_ms < oldest_timeline_ms) {
        asd_timeline_.pop_front();
      }
      if (should_score || !vad_gate_open) {
        std::lock_guard<std::mutex> lock(state_mu_);
        current_asd_scores_ = scores;
      }

      std::vector<FusionEvent> events;
      for (const auto& utterance : input.utterances) {
        FusionInput utterance_input = input;
        utterance_input.utterances = {utterance};
        utterance_input.active_speaker_scores = AggregateAsdScoresForUtterance(
            utterance, asd_timeline_, scores, config_.asd.speaking_threshold, interval_ms);
        auto utterance_events =
            BuildFusionEvents(utterance_input, fusion_->Fuse(utterance_input));
        events.insert(events.end(), utterance_events.begin(), utterance_events.end());
      }

      // Apply diarization speaker ID resolution and update affinity matrix
      for (auto& ev : events) {
        if (ev.utterance_id.empty()) continue; // Skip pure track updates with no ASR text

        // Find the matching utterance in the input to get voice_spk_id and is_overlap
        std::string voice_spk_id;
        bool is_overlap = false;
        for (const auto& utt : input.utterances) {
          if (utt.utterance_id == ev.utterance_id) {
            voice_spk_id = utt.voice_spk_id;
            is_overlap = utt.is_overlap;
            break;
          }
        }
        ev.voice_spk_id = voice_spk_id;

        if (voice_spk_id.empty()) {
          // If no voice speaker id (e.g. ASR transcript with no diarization), fallback
          if (is_overlap) {
            ev.speaker_id = "overlap";
            ev.speaker_name = "多人";
          } else if (!ev.person_track_ids.empty()) {
            ev.speaker_id = "P" + std::to_string(ev.person_track_ids.front());
            ev.speaker_name = ev.speaker_id;
          } else if (ev.position == "ambiguous") {
            ev.speaker_id = "ambiguous";
            ev.speaker_name = "待确认";
          } else {
            ev.speaker_id = "offscreen";
            ev.speaker_name = "画外";
          }
          continue;
        }

        // Update voice-track affinity if fusion successfully associated this utterance with a track
        if (ev.final && !ev.tentative && !is_overlap &&
            ev.person_track_ids.size() == 1 &&
            ev.confidence >= config_.asd.speaking_threshold && diarization_) {
          int matched_pid = ev.person_track_ids.front();
          diarization_->UpdateAffinity(voice_spk_id, matched_pid, ev.confidence, input.tracks);
        }

        // Resolve speaker ID using the updated affinity matrix
        if (ev.position == "ambiguous") {
          ev.speaker_id = voice_spk_id;
          ev.speaker_name = voice_spk_id;
        } else if (diarization_) {
          ev.speaker_id = diarization_->ResolveSpeaker(voice_spk_id, input.tracks);
          ev.speaker_name = ev.speaker_id; // Will be mapped to gallery name in gateway server
        }

        if (is_overlap) {
          ev.speaker_id = "overlap";
          ev.speaker_name = "多人";
        }
      }
      for (const auto& ev : events) {
        if (!ev.final) {
          continue;
        }
        std::ostringstream event_record;
        event_record << "{\"type\":\"fusion_final\",\"utterance_id\":\""
                     << EscapeJsonString(ev.utterance_id) << "\",\"start_ms\":"
                     << ev.start_ms << ",\"end_ms\":" << ev.end_ms
                     << ",\"position\":\"" << EscapeJsonString(ev.position)
                     << "\",\"confidence\":" << ev.confidence
                     << ",\"decision_reason\":\""
                     << EscapeJsonString(ev.decision_reason) << "\",\"voice_spk_id\":\""
                     << EscapeJsonString(ev.voice_spk_id) << "\",\"speaker_id\":\""
                     << EscapeJsonString(ev.speaker_id) << "\"}";
        RecordSessionLine(event_record.str());
      }

      if (events.empty() && !input.tracks.empty() && should_score) {
        FusionEvent track_event;
        track_event.event_type = "asd_update";
        track_event.utterance_id = "";
        track_event.start_ms = 0;
        track_event.end_ms = 0;
        track_event.text = "";
        track_event.final = false;
        track_event.position = "offscreen";
        track_event.confidence = 0.0f;
        track_event.tentative = true;
        track_event.decision_reason = "asd_track_update";

        for (const auto& track : input.tracks) {
          float p_active = 0.0F;
          for (const auto& score : scores) {
            if (score.person_track_id == track.person_track_id) {
              p_active = std::max(p_active, score.p_active);
            }
          }
          track_event.tracks.push_back(MakeTrackView(track, p_active));
        }
        events.push_back(track_event);
      }

      if (!events.empty()) {
        EnqueueEvents(std::move(events));
      }
    } catch (const std::exception& e) {
      SetLatestError(std::string("fusion pipeline: ") + e.what());
      std::cerr << "Error in FusionThreadLoop: " << e.what() << "\n";
    }
  }
}

} // namespace speaker_id
