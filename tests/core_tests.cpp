#include "speaker_id/core/config.hpp"
#include "speaker_id/core/audio_clock.hpp"
#include "speaker_id/core/llm_turn_gate.hpp"
#include "speaker_id/core/pipeline.hpp"
#include "speaker_id/modules/asd.hpp"
#include "speaker_id/modules/diarization.hpp"
#include "speaker_id/modules/face.hpp"
#include "speaker_id/modules/kalman_tracker.hpp"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    ++g_failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

void ExpectEq(const std::string& actual, const std::string& expected, const std::string& message) {
  if (actual != expected) {
    ++g_failures;
    std::cerr << "FAIL: " << message << " expected='" << expected << "' actual='" << actual << "'\n";
  }
}

void ExpectEq(int actual, int expected, const std::string& message) {
  if (actual != expected) {
    ++g_failures;
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << "\n";
  }
}

void ExpectNear(float actual, float expected, float tolerance, const std::string& message) {
  if (std::fabs(actual - expected) > tolerance) {
    ++g_failures;
    std::cerr << "FAIL: " << message << " expected=" << expected << " actual=" << actual << "\n";
  }
}

speaker_id::TrackEvent MakeTrack(int person_id, float x1, float x2) {
  speaker_id::TrackEvent track;
  track.person_track_id = person_id;
  track.face_track_id = person_id + 100;
  track.bbox = speaker_id::BBox{x1, 100.0F, x2, 600.0F};
  track.quality = speaker_id::Quality::kGood;
  track.confidence = 0.9F;
  track.track_state = "tracked";
  return track;
}

speaker_id::ActiveSpeakerScore MakeScore(int person_id, float p_active) {
  speaker_id::ActiveSpeakerScore score;
  score.person_track_id = person_id;
  score.face_track_id = person_id + 100;
  score.timestamp_ms = 1200;
  score.p_active = p_active;
  score.p_audible_speaking = p_active;
  score.p_av_sync = p_active;
  score.face_quality = 0.9F;
  return score;
}

speaker_id::UtteranceEvent MakeUtterance(bool final = true) {
  speaker_id::UtteranceEvent utterance;
  utterance.utterance_id = "utt_test";
  utterance.start_ms = 1000;
  utterance.end_ms = 1600;
  utterance.text = "hello";
  utterance.final = final;
  utterance.confidence = 0.8F;
  return utterance;
}

std::vector<speaker_id::SpeakerAttribution> FuseScores(
    const std::vector<speaker_id::ActiveSpeakerScore>& scores) {
  speaker_id::RuleFusionBackend fusion;
  speaker_id::FusionInput input;
  input.tracks = {MakeTrack(1, 100.0F, 260.0F), MakeTrack(2, 840.0F, 1000.0F)};
  input.vad_segments = {speaker_id::VadEvent{1000, 1600, 0.9F}};
  input.utterances = {MakeUtterance()};
  input.active_speaker_scores = scores;
  return fusion.Fuse(input);
}

void ConfigLoadsOrinDefaults() {
  const auto config = speaker_id::LoadConfig("configs/orin_nx.yaml");
  ExpectEq(config.name, "video-speaker-id", "config app name");
  ExpectEq(config.profile, "orin_nx_16gb_25w", "config app profile");
  ExpectEq(config.runtime.target_latency_ms, 500, "target latency");
  ExpectEq(config.runtime.worker_threads, 6, "worker threads");
  Expect(config.runtime.use_cuda, "cuda flag");
  Expect(config.runtime.use_tensorrt, "tensorrt flag");
  Expect(!config.runtime.allow_mock_inputs, "mock input fallback disabled");
  ExpectEq(config.asd.backend, "lr_asd_onnx", "C++ ASD backend");
  ExpectNear(config.asd.speaking_threshold, 0.55F, 0.001F, "ASD confirmation threshold");
  ExpectNear(config.fusion.visible_speaker_threshold, 0.35F, 0.001F,
             "visible candidate threshold");
  ExpectEq(config.services.at("audio").port, 7100, "audio port");
  ExpectEq(config.services.at("video").port, 7200, "video port");
  ExpectEq(config.services.at("fusion").port, 7400, "fusion port");
  ExpectEq(config.services.at("api").port, 8080, "api port");
}

void ConfigMissingFileThrows() {
  bool threw = false;
  try {
    (void)speaker_id::LoadConfig("configs/does-not-exist.yaml");
  } catch (const std::exception&) {
    threw = true;
  }
  Expect(threw, "missing config throws");
}

void ConfigLoadsMacPhaseTwoPolicy() {
  const auto config = speaker_id::LoadConfig("configs/live_mac.yaml");
  ExpectEq(config.runtime.frame_queue_size, 2, "Mac latest-frame capture queue");
  ExpectEq(config.face.run_every_n_frames, 4, "Mac SCRFD cadence");
  ExpectEq(config.face.hold_ms, 600, "Mac projected face hold");
  ExpectEq(config.face.asd_crop_ttl_ms, 1200, "Mac ASD crop TTL");
  ExpectEq(config.face.tracked_asd_crop_ttl_ms, 2600,
           "Mac person-backed ASD crop TTL");
  ExpectEq(config.video.tracker_max_age_frames, 90,
           "Mac internal tracker occlusion memory");
  ExpectEq(config.face.identity_embedding_interval_ms, 750, "Mac ArcFace cadence");
  ExpectEq(config.face.asd_crop_max_geometry_age_ms, 650,
           "Mac ASD crop geometry freshness");
  ExpectEq(config.face.asd_crop_edge_margin_px, 12, "Mac ASD crop edge margin");
  ExpectNear(config.face.asd_crop_min_visible_ratio, 0.90F, 0.001F,
             "Mac ASD crop visible ratio");
  ExpectEq(config.preview.width, 960, "Mac low-latency preview width");
  ExpectEq(config.preview.height, 540, "Mac low-latency preview height");
  ExpectEq(config.preview.fps, 15, "Mac low-latency preview cadence");
  ExpectEq(config.preview.jpeg_quality, 75, "Mac low-latency JPEG quality");
  ExpectEq(config.asd.vad_hangover_ms, 500, "Mac ASD VAD hangover");
  ExpectEq(config.asd.min_crop_count, 20, "Mac ASD minimum crop coverage");
  ExpectEq(config.asd.max_crop_gap_ms, 120, "Mac ASD maximum crop gap");
  ExpectEq(config.asd.max_candidate_tracks, 4, "Mac ASD bounded candidates");
  Expect(config.diagnostics.session_replay_enabled, "Mac LR-ASD replay enabled");
  ExpectNear(config.fusion.offscreen_threshold, 0.25F, 0.001F,
             "Mac offscreen threshold");
  ExpectNear(config.fusion.visible_speaker_threshold, 0.35F, 0.001F,
             "Mac tentative visible threshold");
  ExpectNear(config.asd.speaking_threshold, 0.55F, 0.001F,
             "Mac confirmed ASD threshold");
}

void ConfigMalformedYamlThrows() {
  const std::string path = "/tmp/speaker_id_bad_config.yaml";
  {
    std::ofstream out(path);
    out << "app:\n  name: broken\nruntime: [\n";
  }
  bool threw = false;
  try {
    (void)speaker_id::LoadConfig(path);
  } catch (const std::exception&) {
    threw = true;
  }
  std::remove(path.c_str());
  Expect(threw, "malformed YAML throws");
}

void FusionDetectsOffscreen() {
  speaker_id::RuleFusionBackend fusion;
  speaker_id::FusionInput input;
  input.vad_segments = {speaker_id::VadEvent{1000, 1600, 0.9F}};
  input.utterances = {MakeUtterance()};
  const auto results = fusion.Fuse(input);
  ExpectEq(static_cast<int>(results.size()), 1, "offscreen result count");
  ExpectEq(results.front().position, "offscreen", "offscreen position");
  ExpectEq(static_cast<int>(results.front().person_track_ids.size()), 0, "offscreen has no visible speaker");
}

void FusionVisibleLowScoresRemainAmbiguous() {
  const auto results = FuseScores({MakeScore(1, 0.20F), MakeScore(2, 0.20F)});
  ExpectEq(results.front().position, "ambiguous",
           "visible track with conflicting ASD evidence is ambiguous");
  ExpectEq(results.front().decision_reason,
           "visible_tracks_with_conflicting_asd_evidence",
           "visible low-score reason is explicit");
}

void FusionDoesNotConfirmThirtyPercentAsd() {
  const auto results = FuseScores({MakeScore(1, 0.30F), MakeScore(2, 0.12F)});
  ExpectEq(results.front().position, "ambiguous", "30 percent ASD remains ambiguous");
  ExpectEq(static_cast<int>(results.front().person_track_ids.size()), 0,
           "30 percent ASD does not bind a visible speaker");
}

void FusionDistinguishesMissingAsdWindow() {
  const auto results = FuseScores({});
  ExpectEq(results.front().position, "ambiguous", "missing ASD window is ambiguous");
  ExpectEq(results.front().decision_reason, "visible_tracks_without_usable_asd_window",
           "missing ASD window reason");
}

void FusionDetectsLeftAndRightFromGeometry() {
  auto left = FuseScores({MakeScore(1, 0.82F), MakeScore(2, 0.20F)});
  ExpectEq(left.front().position, "left", "left position");
  ExpectEq(left.front().person_track_ids.front(), 1, "left id");

  auto right = FuseScores({MakeScore(1, 0.20F), MakeScore(2, 0.83F)});
  ExpectEq(right.front().position, "right", "right position");
  ExpectEq(right.front().person_track_ids.front(), 2, "right id");
}

void FusionAllowsMultiLabelOverlap() {
  const auto results = FuseScores({MakeScore(1, 0.82F), MakeScore(2, 0.79F)});
  ExpectEq(results.front().position, "overlap", "overlap position");
  ExpectEq(static_cast<int>(results.front().person_track_ids.size()), 2, "overlap speaker count");
  ExpectNear(results.front().confidence, 0.82F, 0.001F, "overlap keeps max confidence");
}

void FusionDoesNotNormalizeAsdScores() {
  const auto results = FuseScores({MakeScore(1, 0.90F), MakeScore(2, 0.85F)});
  ExpectEq(results.front().position, "overlap", "non-normalized overlap position");
  ExpectEq(static_cast<int>(results.front().person_track_ids.size()), 2, "non-normalized speaker count");
  ExpectNear(results.front().confidence, 0.90F, 0.001F, "non-normalized keeps raw max confidence");
}

void FusionMarksTentativeUtterance() {
  speaker_id::RuleFusionBackend fusion;
  speaker_id::FusionInput input;
  input.tracks = {MakeTrack(1, 100.0F, 260.0F)};
  input.vad_segments = {speaker_id::VadEvent{1000, 1600, 0.9F}};
  input.utterances = {MakeUtterance(false)};
  input.active_speaker_scores = {MakeScore(1, 0.80F)};
  const auto results = fusion.Fuse(input);
  Expect(results.front().tentative, "tentative flag follows utterance finality");
}

void FusionRequiresVadEvidence() {
  speaker_id::RuleFusionBackend fusion;
  speaker_id::FusionInput input;
  input.tracks = {MakeTrack(1, 100.0F, 260.0F)};
  input.utterances = {MakeUtterance()};
  input.active_speaker_scores = {MakeScore(1, 0.80F)};
  const auto results = fusion.Fuse(input);
  ExpectEq(static_cast<int>(results.size()), 0, "fusion requires VAD evidence");
}

void BuildFusionEventsSkipsUnknownUtterance() {
  speaker_id::FusionInput input;
  input.tracks = {MakeTrack(1, 100.0F, 260.0F)};
  input.active_speaker_scores = {MakeScore(1, 0.80F)};

  speaker_id::SpeakerAttribution attribution;
  attribution.utterance_id = "missing";
  attribution.position = "left";
  attribution.person_track_ids = {1};
  attribution.confidence = 0.80F;
  attribution.tentative = false;

  const auto events = speaker_id::BuildFusionEvents(input, {attribution});
  ExpectEq(static_cast<int>(events.size()), 0, "unknown utterance is skipped");
}

void TrackerIdenticalBoxesIouOne() {
  speaker_id::BBox box{0.0F, 0.0F, 100.0F, 100.0F};
  ExpectNear(speaker_id::ComputeIou(box, box), 1.0F, 0.001F, "IoU identical boxes");
}

void TrackerNoOverlapIouZero() {
  speaker_id::BBox a{0.0F, 0.0F, 50.0F, 50.0F};
  speaker_id::BBox b{100.0F, 100.0F, 200.0F, 200.0F};
  ExpectNear(speaker_id::ComputeIou(a, b), 0.0F, 0.001F, "IoU no overlap");
}

void TrackerPartialOverlap() {
  speaker_id::BBox a{0.0F, 0.0F, 100.0F, 100.0F};
  speaker_id::BBox b{50.0F, 50.0F, 150.0F, 150.0F};
  float expected = 2500.0F / 17500.0F;
  ExpectNear(speaker_id::ComputeIou(a, b), expected, 0.001F, "IoU partial overlap");
}

void TrackerCreatesNewTrack() {
  speaker_id::SimplePersonTracker tracker;
  std::vector<speaker_id::BBox> boxes = {speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}};
  auto tracks = tracker.Update(boxes, 1280, 720);
  ExpectEq(static_cast<int>(tracks.size()), 1, "Creates track");
  ExpectEq(tracks[0].person_track_id, 1, "First ID");
}

void TrackerMatchesExistingTrack() {
  speaker_id::SimplePersonTracker tracker;
  tracker.Update({speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}}, 1280, 720);
  auto tracks = tracker.Update({speaker_id::BBox{105.0F, 105.0F, 305.0F, 505.0F}}, 1280, 720);
  ExpectEq(static_cast<int>(tracks.size()), 1, "Matches track");
  ExpectEq(tracks[0].person_track_id, 1, "ID matches");
}

void TrackerRecoversFastMotionWithCenterDistance() {
  speaker_id::SimplePersonTracker tracker;
  tracker.Update({speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}}, 1280, 720);
  const auto tracks =
      tracker.Update({speaker_id::BBox{270.0F, 100.0F, 470.0F, 500.0F}}, 1280, 720);
  ExpectEq(static_cast<int>(tracks.size()), 1, "fast motion keeps one person track");
  ExpectEq(tracks.front().person_track_id, 1,
           "center-distance recovery preserves person id");
}

void TrackerRemovesStaleTrack() {
  speaker_id::SimplePersonTracker tracker(0.0F, 0.0F); // max_age_s = 0, render_grace_s = 0
  tracker.Update({speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}}, 1280, 720);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  auto tracks = tracker.Update({speaker_id::BBox{900.0F, 100.0F, 1100.0F, 500.0F}}, 1280, 720);
  ExpectEq(static_cast<int>(tracks.size()), 1, "Stale removed");
  ExpectEq(tracks[0].person_track_id, 2, "New ID after stale");
}

void TrackerLowConfidenceOnlySalvagesExistingTracks() {
  speaker_id::SimplePersonTracker tracker(1.2F, 0.35F, 0.5F, 0.1F, 0.2F);
  auto tracks = tracker.Update(
      {speaker_id::PersonDetection{speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}, 0.8F}},
      1280, 720);
  ExpectEq(static_cast<int>(tracks.size()), 1, "high confidence creates track");
  tracks = tracker.Update(
      {speaker_id::PersonDetection{speaker_id::BBox{105.0F, 105.0F, 305.0F, 505.0F}, 0.2F}},
      1280, 720);
  ExpectEq(static_cast<int>(tracks.size()), 1, "low confidence salvages known track");
  ExpectEq(tracks.front().person_track_id, 1, "salvaged track preserves ID");

  speaker_id::SimplePersonTracker fresh_tracker(1.2F, 0.35F, 0.5F, 0.1F, 0.2F);
  tracks = fresh_tracker.Update(
      {speaker_id::PersonDetection{speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}, 0.2F}},
      1280, 720);
  ExpectEq(static_cast<int>(tracks.size()), 0, "low confidence cannot create new track");
}

void TrackerWaitsForMinimumHits() {
  speaker_id::SimplePersonTracker tracker(1.2F, 0.35F, 0.5F, 0.1F, 0.2F, 3);
  const auto detection =
      speaker_id::PersonDetection{speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}, 0.8F};
  ExpectEq(static_cast<int>(tracker.Update({detection}, 1280, 720).size()), 0,
           "first hit remains tentative");
  ExpectEq(static_cast<int>(tracker.Update({detection}, 1280, 720).size()), 0,
           "second hit remains tentative");
  ExpectEq(static_cast<int>(tracker.Update({detection}, 1280, 720).size()), 1,
           "third hit renders stable track");
}

void TrackerPredictOnlyDoesNotDegradeScheduledSkip() {
  speaker_id::SimplePersonTracker tracker;
  auto tracks = tracker.Update({speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}}, 1280, 720);
  ExpectEq(tracks.front().track_state, "tracked", "observed person starts tracked");
  const auto quality = tracks.front().quality;
  tracks = tracker.PredictOnly(1280, 720);
  ExpectEq(tracks.front().track_state, "tracked", "scheduled skip stays tracked");
  Expect(tracks.front().quality == quality, "scheduled skip preserves quality");
}

void TrackerRealMissTransitionsToPredicted() {
  speaker_id::SimplePersonTracker tracker;
  tracker.Update({speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}}, 1280, 720);
  const auto tracks = tracker.Update(std::vector<speaker_id::PersonDetection>{}, 1280, 720);
  ExpectEq(tracks.front().track_state, "predicted", "real detector miss becomes predicted");
  Expect(tracks.front().quality == speaker_id::Quality::kLow,
         "real detector miss degrades quality");
}

void TrackerDoesNotInventFaceIdentity() {
  speaker_id::SimplePersonTracker tracker;
  const auto tracks =
      tracker.Update({speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}}, 1280, 720);
  Expect(!tracks.front().face_track_id.has_value(), "person tracker does not invent face id");
}

void TrackerAcceptsObservedIdentityHint() {
  speaker_id::SimplePersonTracker tracker;
  auto tracks = tracker.Update({speaker_id::BBox{100.0F, 100.0F, 300.0F, 500.0F}}, 1280, 720);
  tracks.front().face_track_id = 7;
  tracks.front().identity_state = "tracking";
  tracker.ApplyIdentityHints(tracks);
  tracks = tracker.PredictOnly(1280, 720);
  Expect(tracks.front().face_track_id == std::optional<int>(7),
         "observed identity hint writes back into person tracker");
}

void LatestFrameQueueDropsOldestFrame() {
  speaker_id::BoundedQueue<int> queue(1);
  Expect(queue.PushOrDrop(1), "latest-frame queue accepts first value");
  Expect(queue.PushOrDrop(2), "latest-frame queue accepts replacement value");
  ExpectEq(static_cast<int>(queue.Size()), 1, "latest-frame queue remains bounded");
  ExpectEq(static_cast<int>(queue.DroppedCount()), 1, "latest-frame queue counts dropped frame");
  ExpectEq(queue.Pop().value_or(-1), 2, "latest-frame queue retains newest frame");
}

void LatestFrameQueueTryPopIsNonBlocking() {
  speaker_id::BoundedQueue<int> queue(1);
  Expect(!queue.TryPop().has_value(), "empty latest-frame queue returns immediately");
  queue.PushOrDrop(7);
  ExpectEq(queue.TryPop().value_or(-1), 7, "try-pop receives queued value");
}

void AudioClockPllCorrectsPipeDrift() {
  speaker_id::AudioMonotonicPll clock(500, 1.0, 8);
  std::int64_t arrival_end_ms = 1040;
  speaker_id::AudioClockStamp stamp;
  for (int index = 0; index < 100; ++index) {
    stamp = clock.Stamp(arrival_end_ms, 40);
    arrival_end_ms += 45;
  }
  Expect(std::llabs(stamp.drift_ms) <= 8,
         "PLL prevents unbounded pipe clock drift");
  Expect(!stamp.reanchored, "bounded drift does not trigger reanchor");
}

void AudioClockPllReanchorsLargeGap() {
  speaker_id::AudioMonotonicPll clock(500, 0.08, 8);
  (void)clock.Stamp(1040, 40);
  const auto stamp = clock.Stamp(3000, 40);
  Expect(stamp.reanchored, "large FFmpeg pipe gap triggers reanchor");
  ExpectEq(static_cast<int>(stamp.timestamp_ms), 2960,
           "reanchor uses observed monotonic chunk start");
}

void FaceBBoxClampRejectsInvalidRoi() {
  speaker_id::BBox invalid{-20.0F, -30.0F, 2.0F, 3.0F};
  Expect(!speaker_id::ClampBBoxToFrame(invalid, 1280, 720, 8),
         "clamp rejects tiny out-of-frame face ROI");
  speaker_id::BBox valid{-20.0F, -30.0F, 80.0F, 90.0F};
  Expect(speaker_id::ClampBBoxToFrame(valid, 1280, 720, 8),
         "clamp keeps usable partially visible face ROI");
  ExpectNear(valid.x1, 0.0F, 0.001F, "clamp moves x1 inside frame");
  ExpectNear(valid.y1, 0.0F, 0.001F, "clamp moves y1 inside frame");
}

void AsdFinalAggregatesWholeUtterance() {
  std::deque<speaker_id::AsdTimelinePoint> timeline = {
      {1100, {MakeScore(1, 0.20F)}},
      {1400, {MakeScore(1, 0.80F)}},
      {1700, {MakeScore(1, 0.70F)}},
      {2000, {MakeScore(1, 0.10F)}},
  };
  auto utterance = MakeUtterance();
  utterance.start_ms = 1000;
  utterance.end_ms = 2000;
  const auto aggregated = speaker_id::AggregateAsdScoresForUtterance(
      utterance, timeline, {MakeScore(1, 0.10F)}, 0.55F, 320);
  ExpectEq(static_cast<int>(aggregated.size()), 1, "final ASD aggregation returns track");
  ExpectNear(aggregated.front().p_active, 0.70F, 0.001F, "final ASD uses turn p75");
  ExpectNear(aggregated.front().mean_p_active, 0.45F, 0.001F, "final ASD exposes mean");
  ExpectNear(aggregated.front().peak_p_active, 0.80F, 0.001F, "final ASD exposes peak");
  ExpectNear(aggregated.front().active_ratio, 0.50F, 0.001F, "final ASD exposes active ratio");
  ExpectEq(aggregated.front().stable_ms, 640, "final ASD exposes stable duration");
}

void LrAsdMfccMatchesPythonSpeechFeatures() {
  constexpr int kSampleRate = 16000;
  std::vector<float> audio(kSampleRate);
  for (int index = 0; index < kSampleRate; ++index) {
    audio[index] =
        0.08F * std::sin(2.0F * 3.14159265358979323846F * 220.0F * index / kSampleRate);
  }
  const auto mfcc = speaker_id::ExtractLrAsdMfcc(audio, kSampleRate);
  const std::vector<float> expected = {
      17.0368919F, 21.5773525F, 12.2625389F, 6.2799191F, 0.1998568F,
      -6.7375870F, -12.8835077F, -18.8733902F, -22.8897781F,
      -25.1604118F, -24.0045967F, -21.8409767F, -17.9486084F,
  };
  ExpectEq(static_cast<int>(mfcc.size()), 100 * 13, "MFCC shape matches LR-ASD graph");
  for (std::size_t index = 0; index < expected.size() && index < mfcc.size(); ++index) {
    ExpectNear(mfcc[index], expected[index], 0.20F,
               "MFCC first frame coefficient " + std::to_string(index));
  }
}

void LlmTurnGateOnlyCommitsFinalizedTurns() {
  speaker_id::LlmTurnGate gate;
  speaker_id::FusionEvent event;
  event.utterance_id = "utt_1";
  event.text = "hello";
  event.speaker_id = "spk_1";
  event.final = false;
  event.tentative = true;
  Expect(gate.Evaluate(event).action == speaker_id::LlmTurnAction::kPrefetch,
         "tentative ASR only permits LLM prefetch");
  event.final = true;
  event.tentative = false;
  Expect(gate.Evaluate(event).action == speaker_id::LlmTurnAction::kCommit,
         "finalized attribution permits LLM commit");
  event.text.clear();
  Expect(gate.Evaluate(event).action == speaker_id::LlmTurnAction::kIgnore,
         "empty ASD updates do not reach LLM");
}

void OrtLrAsdScoresSynchronizedTrackWindow() {
  const std::string model_path = "models/asd/lr-asd/lr_asd_talkset.onnx";
  std::ifstream model(model_path);
  if (!model) {
    std::cout << "SKIP: LR-ASD ONNX smoke model is not installed\n";
    return;
  }
  speaker_id::OrtLrAsdBackend backend(model_path);
  speaker_id::AsdInputWindow window;
  window.start_ms = 1000;
  window.end_ms = 2000;
  window.sample_rate = 16000;
  window.speech_active = true;
  window.tracks = {MakeTrack(1, 100.0F, 260.0F)};
  window.tracks.front().face.quality_score = 0.9F;
  for (int index = 0; index < window.sample_rate; ++index) {
    window.audio_samples.push_back(
        0.08F * std::sin(2.0F * 3.14159265358979323846F * 220.0F * index /
                         window.sample_rate));
  }
  for (int index = 0; index < 25; ++index) {
    speaker_id::AsdVisualSample sample;
    sample.timestamp_ms = 1000 + index * 1000 / 24;
    sample.grayscale_112.assign(112 * 112, 96.0F + static_cast<float>(index % 5));
    window.visual_samples[1].push_back(std::move(sample));
  }
  const auto scores = backend.ScoreWindow(window);
  ExpectEq(static_cast<int>(scores.size()), 1, "ORT LR-ASD scores synchronized track");
  if (!scores.empty()) {
    Expect(std::isfinite(scores.front().p_active), "ORT LR-ASD score is finite");
    Expect(scores.front().p_active >= 0.0F && scores.front().p_active <= 1.0F,
           "ORT LR-ASD score stays within probability bounds");
  }
}

void OrtDiarizationAcceptsLongWindowAndModelMetadata() {
  const std::string segmentation_model = "models/diarization/pyannote-segmentation-3.0.onnx";
  const std::string voiceprint_model =
      "models/voiceprint/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
  if (!std::ifstream(segmentation_model) || !std::ifstream(voiceprint_model)) {
    std::cout << "SKIP: diarization smoke models are not installed\n";
    return;
  }
  speaker_id::OrtDiarizationBackend backend(
      segmentation_model, voiceprint_model, 0.50F, 200);
  std::vector<float> silence(160001, 0.0F);
  const auto segments = backend.ProcessUtterance(silence, 16000);
  Expect(!segments.empty(), "diarization returns fallback for a long silent window");
  for (const auto& segment : segments) {
    Expect(segment.left_sample >= 0, "diarization segment begins inside audio");
    Expect(segment.right_sample <= static_cast<int>(silence.size()),
           "diarization segment ends inside audio");
    Expect(segment.right_sample > segment.left_sample,
           "diarization segment has positive duration");
  }
}

}  // namespace

int main() {
  ConfigLoadsOrinDefaults();
  ConfigLoadsMacPhaseTwoPolicy();
  ConfigMissingFileThrows();
  ConfigMalformedYamlThrows();
  FusionDetectsOffscreen();
  FusionVisibleLowScoresRemainAmbiguous();
  FusionDoesNotConfirmThirtyPercentAsd();
  FusionDistinguishesMissingAsdWindow();
  FusionDetectsLeftAndRightFromGeometry();
  FusionAllowsMultiLabelOverlap();
  FusionDoesNotNormalizeAsdScores();
  FusionMarksTentativeUtterance();
  FusionRequiresVadEvidence();
  BuildFusionEventsSkipsUnknownUtterance();
  TrackerIdenticalBoxesIouOne();
  TrackerNoOverlapIouZero();
  TrackerPartialOverlap();
  TrackerCreatesNewTrack();
  TrackerMatchesExistingTrack();
  TrackerRecoversFastMotionWithCenterDistance();
  TrackerRemovesStaleTrack();
  TrackerLowConfidenceOnlySalvagesExistingTracks();
  TrackerWaitsForMinimumHits();
  TrackerPredictOnlyDoesNotDegradeScheduledSkip();
  TrackerRealMissTransitionsToPredicted();
  TrackerDoesNotInventFaceIdentity();
  TrackerAcceptsObservedIdentityHint();
  LatestFrameQueueDropsOldestFrame();
  LatestFrameQueueTryPopIsNonBlocking();
  AudioClockPllCorrectsPipeDrift();
  AudioClockPllReanchorsLargeGap();
  FaceBBoxClampRejectsInvalidRoi();
  AsdFinalAggregatesWholeUtterance();
  LrAsdMfccMatchesPythonSpeechFeatures();
  LlmTurnGateOnlyCommitsFinalizedTurns();
  OrtLrAsdScoresSynchronizedTrackWindow();
  OrtDiarizationAcceptsLongWindowAndModelMetadata();

  if (g_failures != 0) {
    std::cerr << g_failures << " regression test(s) failed\n";
    return 1;
  }
  std::cout << "speaker_id_core_tests passed\n";
  return 0;
}
