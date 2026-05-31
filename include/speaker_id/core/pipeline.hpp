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

  // Retrieve all recent fusion events (clears internal cache after reading)
  std::vector<FusionEvent> GetLatestEvents();

  // Retrieve thread-safe copy of current tracks
  std::vector<TrackEvent> GetCurrentTracks();
  std::string GetLatestError();

  FaceEngine* GetFaceEngine() const { return face_engine_.get(); }
  DiarizationBackend* GetDiarizationEngine() const { return diarization_.get(); }

 private:
  void VideoThreadLoop();
  void AudioThreadLoop();
  void FusionThreadLoop();
  void SetLatestError(const std::string& error);

  AppConfig config_;
  std::shared_ptr<VisionBackend> vision_;
  std::shared_ptr<VadBackend> vad_;
  std::shared_ptr<AsrBackend> asr_;
  std::shared_ptr<AsdBackend> asd_;
  std::shared_ptr<FusionBackend> fusion_;
  std::shared_ptr<DiarizationBackend> diarization_;

  BoundedQueue<FrameEvent> frame_queue_;
  BoundedQueue<AudioChunkEvent> audio_queue_;

  std::thread video_thread_;
  std::thread audio_thread_;
  std::thread fusion_thread_;

  std::atomic<bool> running_{false};
  
  std::mutex events_mu_;
  std::vector<FusionEvent> latest_events_;
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

  SimplePersonTracker tracker_;
  std::unique_ptr<FaceEngine> face_engine_;

  // Audio collector buffer matching Python experience
  std::vector<float> current_speech_samples_;
  std::deque<std::vector<float>> preroll_buffer_;
};

} // namespace speaker_id
