#include "speaker_id/core/pipeline.hpp"

#include <chrono>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cmath>

namespace speaker_id {
namespace {

std::int64_t MonotonicNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<float> ExtractAsdCrop(const cv::Mat& frame, const TrackEvent& track) {
  if (frame.empty()) {
    return {};
  }
  const auto& box = track.face.bbox;
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
  if (x2 <= x1 || y2 <= y1) {
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
      audio_queue_(512),
      tracker_(
          static_cast<float>(config_.video.tracker_max_age_frames) /
              static_cast<float>(std::max(1, config_.video.fps)),
          0.35F,
          config_.video.tracker_high_confidence_threshold,
          config_.video.tracker_low_confidence_threshold,
          config_.video.tracker_iou_threshold,
          config_.video.tracker_min_hits) {
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

  video_thread_ = std::thread(&StreamingPipeline::VideoThreadLoop, this);
  audio_thread_ = std::thread(&StreamingPipeline::AudioThreadLoop, this);
  fusion_thread_ = std::thread(&StreamingPipeline::FusionThreadLoop, this);

  return true;
}

void StreamingPipeline::Stop() {
  if (!running_.exchange(false)) {
    return;
  }

  frame_queue_.Close();
  audio_queue_.Close();

  if (video_thread_.joinable()) {
    video_thread_.join();
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
  std::vector<FusionEvent> events = std::move(latest_events_);
  latest_events_.clear();
  return events;
}

std::vector<TrackEvent> StreamingPipeline::GetCurrentTracks() {
  std::lock_guard<std::mutex> lock(state_mu_);
  return current_tracks_;
}

std::string StreamingPipeline::GetLatestError() {
  std::lock_guard<std::mutex> lock(error_mu_);
  return latest_error_;
}

void StreamingPipeline::SetLatestError(const std::string& error) {
  std::lock_guard<std::mutex> lock(error_mu_);
  latest_error_ = error;
}


void StreamingPipeline::VideoThreadLoop() {
  int frame_count = 0;
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
        auto raw_detections = vision_->AcceptFrame(*frame_opt);
        std::vector<PersonDetection> detections;
        detections.reserve(raw_detections.size());
        for (const auto& det : raw_detections) {
          detections.push_back(PersonDetection{det.bbox, det.confidence});
        }
        tracks = tracker_.Update(detections, frame_opt->width, frame_opt->height);
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
        // Feed empty detections to Kalman tracker so it predicts smooth movements in skipped frames
        tracks = tracker_.Update(std::vector<PersonDetection>{}, frame_opt->width, frame_opt->height);
      }

      // Decode frame image for FaceEngine
      cv::Mat frame_mat;
      if (frame_opt->data.size() > 4 && frame_opt->data[0] == 0xFF && frame_opt->data[1] == 0xD8) {
        frame_mat = cv::imdecode(frame_opt->data, cv::IMREAD_COLOR);
      } else if (frame_opt->width > 0 && frame_opt->height > 0 && frame_opt->data.size() == static_cast<size_t>(frame_opt->width * frame_opt->height * 3)) {
        frame_mat = cv::Mat(frame_opt->height, frame_opt->width, CV_8UC3, const_cast<uint8_t*>(frame_opt->data.data())).clone();
      }

      if (!frame_mat.empty() && face_engine_) {
        tracks = face_engine_->UpdateTracks(frame_mat, tracks, frame_count, frame_opt->timestamp_ms);
      }

      frame_count++;

      {
        std::lock_guard<std::mutex> lock(state_mu_);
        current_tracks_ = tracks;
        const auto visual_timestamp_ms =
            frame_opt->timestamp_ms + static_cast<std::int64_t>(
                                          std::llround(config_.sync.video_time_offset_ms));
        const auto oldest_visual_ms =
            visual_timestamp_ms - std::max(4000, config_.asd.window_ms + 1000);
        for (const auto& track : tracks) {
          const bool stale_face_allowed =
              !track.face.face_bbox_observed &&
              track.face.face_bbox_last_observed_ms > 0 &&
              frame_opt->timestamp_ms - track.face.face_bbox_last_observed_ms <= 1200;
          if (!track.face.face_bbox_observed && !stale_face_allowed) {
            continue;
          }
          auto crop = ExtractAsdCrop(frame_mat, track);
          if (!crop.empty()) {
            asd_visual_history_[track.person_track_id].push_back(
                AsdVisualSample{visual_timestamp_ms, std::move(crop)});
          }
        }
        for (auto& [_, history] : asd_visual_history_) {
          while (!history.empty() && history.front().timestamp_ms < oldest_visual_ms) {
            history.pop_front();
          }
        }
      }
    } catch (const std::exception& e) {
      SetLatestError(std::string("video pipeline: ") + e.what());
      std::cerr << "Error in VideoThreadLoop: " << e.what() << "\n";
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
        asd_audio_history_.push_back(*audio_opt);
        latest_audio_end_ms_ = audio_opt->timestamp_ms + audio_opt->duration_ms;
        speech_active_ = is_speech_active;
        const auto oldest_audio_ms =
            latest_audio_end_ms_ - std::max(4000, config_.asd.window_ms + 1000);
        while (!asd_audio_history_.empty() &&
               asd_audio_history_.front().timestamp_ms +
                       asd_audio_history_.front().duration_ms <
                   oldest_audio_ms) {
          asd_audio_history_.pop_front();
        }
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
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      input.tracks = current_tracks_;
      input.vad_segments = std::move(current_vad_segments_);
      input.utterances = std::move(current_utterances_);
      current_vad_segments_.clear();
      current_utterances_.clear();
      window.end_ms = latest_audio_end_ms_;
      window.start_ms = std::max<std::int64_t>(0, window.end_ms - config_.asd.window_ms);
      window.sample_rate = config_.audio.sample_rate;
      window.speech_active = speech_active_;
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

    if (input.tracks.empty() && input.vad_segments.empty()) {
      continue;
    }

    try {
      // Perform active speaker scoring
      window.tracks = input.tracks;
      window.vad_segments = input.vad_segments;

      auto scores = asd_->ScoreWindow(window);
      input.active_speaker_scores = scores;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        current_asd_scores_ = scores;
      }

      // Run multi-modal fusion
      auto attributions = fusion_->Fuse(input);
      auto events = BuildFusionEvents(input, attributions);

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

        if (voice_spk_id.empty()) {
          // If no voice speaker id (e.g. ASR transcript with no diarization), fallback
          if (is_overlap) {
            ev.speaker_id = "overlap";
            ev.speaker_name = "多人";
          } else if (!ev.person_track_ids.empty()) {
            ev.speaker_id = "P" + std::to_string(ev.person_track_ids.front());
            ev.speaker_name = ev.speaker_id;
          } else {
            ev.speaker_id = "offscreen";
            ev.speaker_name = "画外";
          }
          continue;
        }

        // Update voice-track affinity if fusion successfully associated this utterance with a track
        if (!is_overlap && ev.person_track_ids.size() == 1 && diarization_) {
          int matched_pid = ev.person_track_ids.front();
          diarization_->UpdateAffinity(voice_spk_id, matched_pid, ev.confidence, input.tracks);
        }

        // Resolve speaker ID using the updated affinity matrix
        if (diarization_) {
          ev.speaker_id = diarization_->ResolveSpeaker(voice_spk_id, input.tracks);
          ev.speaker_name = ev.speaker_id; // Will be mapped to gallery name in gateway server
        }

        if (is_overlap) {
          ev.speaker_id = "overlap";
          ev.speaker_name = "多人";
        }
      }

      if (events.empty() && !input.tracks.empty()) {
        FusionEvent track_event;
        track_event.utterance_id = "";
        track_event.start_ms = 0;
        track_event.end_ms = 0;
        track_event.text = "";
        track_event.final = false;
        track_event.position = "offscreen";
        track_event.confidence = 0.0f;
        track_event.tentative = true;

        for (const auto& track : input.tracks) {
          FusionTrackView view;
          view.person_track_id = track.person_track_id;
          view.face_track_id = track.face_track_id;
          view.bbox = track.bbox;
          view.quality = track.quality;
          view.p_active = 0.0f;
          for (const auto& score : scores) {
            if (score.person_track_id == track.person_track_id) {
              view.p_active = std::max(view.p_active, score.p_active);
            }
          }
          
          view.face = track.face;
          view.body = track.body;
          view.render = track.render;
          view.identity_id = track.identity_id;
          view.identity_name = track.identity_name;
          view.identity_state = track.identity_state;
          view.identity_confidence = track.identity_confidence;

          track_event.tracks.push_back(view);
        }
        events.push_back(track_event);
      }

      if (!events.empty()) {
        std::lock_guard<std::mutex> lock(events_mu_);
        latest_events_.insert(latest_events_.end(), events.begin(), events.end());
      }
    } catch (const std::exception& e) {
      SetLatestError(std::string("fusion pipeline: ") + e.what());
      std::cerr << "Error in FusionThreadLoop: " << e.what() << "\n";
    }
  }
}

} // namespace speaker_id
