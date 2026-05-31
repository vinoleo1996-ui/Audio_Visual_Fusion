#pragma once

#include "speaker_id/api/asr.hpp"
#include "speaker_id/core/config.hpp"

#include <memory>
#include <string>
#include <vector>

#include "speaker_id/api/diarization.hpp"

namespace speaker_id {

class SherpaAsrBackend : public AsrEngine {
  public:
   SherpaAsrBackend(const AsrConfig& config, int cpu_threads = 4);
   ~SherpaAsrBackend() override;
 
   AsrResult Transcribe(const std::vector<float>& samples, int sample_rate) override;
 
   void BeginStreaming() override;
   AsrResult StreamAudio(const std::vector<float>& samples, int sample_rate) override;
   AsrResult EndStreaming() override;
 
   // Audio buffering and VAD segment transcription helpers
   void PushAudio(const AudioChunkEvent& chunk);
   std::vector<UtteranceEvent> AcceptSpeechSegment(const VadEvent& seg, const std::vector<float>& speech_samples, DiarizationEngine* diarization = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace speaker_id
