#pragma once

#include "speaker_id/core/config.hpp"
#include "speaker_id/core/module.hpp"
#include "speaker_id/core/types.hpp"
#include "speaker_id/modules/asd.hpp"
#include "speaker_id/modules/asr.hpp"
#include "speaker_id/modules/fusion.hpp"
#include "speaker_id/modules/vad.hpp"
#include "speaker_id/modules/face.hpp"
#include "speaker_id/modules/vision.hpp"
#include "speaker_id/modules/kalman_tracker.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "speaker_id/api/diarization.hpp"

#include <deque>

namespace speaker_id {

using VisionBackend = VisionEngine;
using VadBackend = VadEngine;
using AsrBackend = SherpaAsrBackend;
using DiarizationBackend = DiarizationEngine;

class StreamingPipeline {
 public:
  StreamingPipeline(
      AppConfig config,
      std::shared_ptr<VisionBackend> vision,
      std::shared_ptr<VadBackend> vad,
      std::shared_ptr<AsrBackend> asr,
      std::shared_ptr<AsdBackend> asd,
      std::shared_ptr<FusionBackend> fusion,
      std::shared_ptr<DiarizationBackend> diarization);
  
  ~StreamingPipeline();

  bool Start();
  void Stop();

  // Thread-safe inputs from external sources
  bool PushFrame(FrameEvent frame);
  bool PushAudio(AudioChunkEvent audio);

  // Compatibility snapshot for local tools. Reading never consumes shared events.
  std::vector<FusionEvent> GetLatestEvents();
  std::vector<FusionEvent> ReadEventsAfter(std::uint64_t event_sequence);
  std::uint64_t LatestEventSequence();

  // Retrieve thread-safe copy of current tracks
  std::vector<TrackEvent> GetCurrentTracks();
  PipelineMetrics GetMetrics();
  std::string GetLatestError();
  void ReportPreviewFrame(double encode_latency_ms, double capture_to_send_ms,
                          std::uint64_t frames_dropped,
                          std::uint64_t ws_dropped_frames,
                          std::uint64_t pipe_frames_dropped,
                          std::size_t pipe_backlog_bytes,
                          const std::string& frame_age_source);

  FaceEngine* GetFaceEngine() const { return face_engine_.get(); }
  DiarizationBackend* GetDiarizationEngine() const { return diarization_.get(); }

 private:
  void VideoThreadLoop();
  void PersonDetectorThreadLoop();
  void FaceThreadLoop();
  void AudioThreadLoop();
  void FusionThreadLoop();
  void SetLatestError(const std::string& error);
  void RecordSessionLine(const std::string& json_line);
  void InitializeSessionReplay();
  void RecordReplayAudio(const AudioChunkEvent& audio);
  void RecordReplayCrop(std::int64_t timestamp_ms, int person_track_id,
                        const std::vector<float>& grayscale_112);
  void RecordReplayWindow(const AsdInputWindow& window,
                          const std::vector<ActiveSpeakerScore>& scores);
  void RecordLatency(std::deque<double>& values, double latency_ms);
  void EnqueueEvents(std::vector<FusionEvent> events);

  AppConfig config_;
  std::shared_ptr<VisionBackend> vision_;
  std::shared_ptr<VadBackend> vad_;
  std::shared_ptr<AsrBackend> asr_;
  std::shared_ptr<AsdBackend> asd_;
  std::shared_ptr<FusionBackend> fusion_;
  std::shared_ptr<DiarizationBackend> diarization_;

  BoundedQueue<FrameEvent> frame_queue_;
  struct PersonDetectorResult {
    std::int64_t capture_timestamp_ms = 0;
    int width = 0;
    int height = 0;
    std::vector<PersonDetection> detections;
  };
  BoundedQueue<FrameEvent> person_detector_queue_;
  BoundedQueue<PersonDetectorResult> person_detector_result_queue_;
  struct FaceFrameJob {
    cv::Mat frame;
    std::vector<TrackEvent> tracks;
    int frame_count = 0;
    std::int64_t capture_timestamp_ms = 0;
  };
  BoundedQueue<FaceFrameJob> face_queue_;
  BoundedQueue<AudioChunkEvent> audio_queue_;

  std::thread video_thread_;
  std::thread person_detector_thread_;
  std::thread face_thread_;
  std::thread audio_thread_;
  std::thread fusion_thread_;

  std::atomic<bool> running_{false};
  
  std::mutex events_mu_;
  std::deque<FusionEvent> event_ring_;
  std::uint64_t next_event_sequence_ = 1;
  std::mutex error_mu_;
  std::string latest_error_;

  // Thread-safe storage of intermediate states for Fusion
  std::mutex state_mu_;
  std::vector<TrackEvent> current_tracks_;
  std::vector<VadEvent> current_vad_segments_;
  std::vector<UtteranceEvent> current_utterances_;
  std::vector<ActiveSpeakerScore> current_asd_scores_;
  std::deque<AudioChunkEvent> asd_audio_history_;
  std::map<int, std::deque<AsdVisualSample>> asd_visual_history_;
  std::int64_t latest_audio_end_ms_ = 0;
  bool speech_active_ = false;
  std::deque<AsdTimelinePoint> asd_timeline_;

  SimplePersonTracker tracker_;
  std::mutex tracker_mu_;
  std::unique_ptr<FaceEngine> face_engine_;

  // Audio collector buffer matching Python experience
  std::vector<float> current_speech_samples_;
  std::deque<std::vector<float>> preroll_buffer_;

  std::mutex metrics_mu_;
  PipelineMetrics metrics_;
  std::deque<double> person_detector_latencies_ms_;
  std::deque<double> face_pipeline_latencies_ms_;
  std::deque<double> asd_latencies_ms_;
  std::deque<double> preview_encode_latencies_ms_;
  std::deque<double> preview_capture_to_send_ms_;
  std::uint64_t asd_windows_total_ = 0;
  std::uint64_t asd_windows_usable_ = 0;
  std::int64_t metrics_started_ms_ = 0;
  std::int64_t last_speech_active_ms_ = 0;
  std::uint64_t next_track_snapshot_sequence_ = 1;
  std::mutex recorder_mu_;
  std::map<int, std::int64_t> replay_last_crop_ms_;
  std::uint64_t replay_audio_bytes_ = 0;
};

} // namespace speaker_id
