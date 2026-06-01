#include "speaker_id/core/config.hpp"

#include <set>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace speaker_id {
namespace {

const YAML::Node Section(const YAML::Node& root, const std::string& key) {
  const auto node = root[key];
  if (!node || !node.IsMap()) {
    throw std::runtime_error("missing or invalid config section: " + key);
  }
  return node;
}

template <typename T>
T Read(const YAML::Node& node, const std::string& key, const T& fallback) {
  const auto value = node[key];
  return value ? value.as<T>() : fallback;
}

void ValidateThreshold(const std::string& name, float value) {
  if (value < 0.0F || value > 1.0F) {
    throw std::runtime_error(name + " must be between 0 and 1");
  }
}

void ValidateTopLevel(const YAML::Node& root) {
  static const std::set<std::string> kKnownSections = {
      "app",      "runtime", "capture", "services", "video",  "face", "audio",
      "vad",      "asr",     "asd", "diarization", "sync", "fusion",
      "timeline", "diagnostics", "ui",
  };
  if (!root || !root.IsMap()) {
    throw std::runtime_error("config root must be a YAML mapping");
  }
  for (const auto& item : root) {
    const auto key = item.first.as<std::string>();
    if (kKnownSections.count(key) == 0) {
      throw std::runtime_error("unknown top-level config section: " + key);
    }
  }
}

void Validate(const AppConfig& config) {
  if (config.runtime.target_latency_ms <= 0 || config.runtime.worker_threads <= 0 ||
      config.runtime.event_ring_size == 0) {
    throw std::runtime_error("runtime latency and worker thread limits must be positive");
  }
  if (config.capture.backend != "ffmpeg_pipe_pll" &&
      config.capture.backend != "libav_pts") {
    throw std::runtime_error("capture.backend must be ffmpeg_pipe_pll or libav_pts");
  }
  if (config.video.width <= 0 || config.video.height <= 0 || config.video.fps <= 0) {
    throw std::runtime_error("video dimensions and fps must be positive");
  }
  if (config.preview.width <= 0 || config.preview.height <= 0 ||
      config.preview.fps <= 0 || config.preview.jpeg_quality <= 0 ||
      config.preview.jpeg_quality > 100 || config.preview.max_frame_age_ms <= 0) {
    throw std::runtime_error("invalid preview dimensions, cadence or quality");
  }
  if (config.video.person_detector_detect_every_n_frames <= 0 ||
      config.video.tracker_max_age_frames <= 0 || config.video.tracker_min_hits <= 0 ||
      config.video.tracker_render_grace_ms < 0) {
    throw std::runtime_error("video detector and tracker cadence must be positive");
  }
  if (config.video.max_detection_result_age_ms <= 0) {
    throw std::runtime_error("video.max_detection_result_age_ms must be positive");
  }
  if (config.face.det_size <= 0 || config.face.run_every_n_frames <= 0 ||
      config.face.hold_ms <= 0 || config.face.asd_crop_ttl_ms < config.face.hold_ms ||
      config.face.tracked_asd_crop_ttl_ms < config.face.asd_crop_ttl_ms ||
      config.face.identity_embedding_interval_ms <= 0) {
    throw std::runtime_error("invalid face detector cadence or hold policy");
  }
  if (config.face.min_bbox_size_px <= 0 || config.face.render_green_hold_ms <= 0 ||
      config.face.render_reacquire_hold_ms < config.face.render_green_hold_ms) {
    throw std::runtime_error("invalid face bbox or render hold policy");
  }
  if (config.face.asd_crop_max_geometry_age_ms <= 0 ||
      config.face.asd_crop_edge_margin_px < 0 ||
      config.face.asd_crop_min_visible_ratio <= 0.0F ||
      config.face.asd_crop_min_visible_ratio > 1.0F) {
    throw std::runtime_error("invalid ASD face crop quality policy");
  }
  if (config.audio.sample_rate <= 0 || config.audio.channels <= 0 ||
      config.audio.chunk_ms <= 0) {
    throw std::runtime_error("audio sample rate, channels and chunk_ms must be positive");
  }
  if (config.vad.min_speech_ms <= 0 || config.vad.min_silence_ms <= 0 ||
      config.vad.max_utterance_ms < config.vad.min_speech_ms) {
    throw std::runtime_error("invalid VAD duration policy");
  }
  if (config.asd.window_ms <= 0 || config.asd.hop_ms <= 0 ||
      config.asd.hop_ms > config.asd.window_ms || config.asd.vad_hangover_ms < 0 ||
      config.asd.min_crop_count <= 0 || config.asd.max_crop_gap_ms <= 0 ||
      config.asd.max_candidate_tracks <= 0) {
    throw std::runtime_error("invalid ASD window policy");
  }
  if (config.sync.max_watermark_lag_ms <= 0 ||
      config.sync.reanchor_threshold_ms <= config.sync.max_watermark_lag_ms ||
      config.sync.pll_alpha < 0.0 || config.sync.pll_alpha > 1.0 ||
      config.sync.max_correction_per_chunk_ms < 0) {
    throw std::runtime_error("invalid sync PLL policy");
  }
  if (config.diagnostics.session_replay_ttl_minutes <= 0) {
    throw std::runtime_error("diagnostics.session_replay_ttl_minutes must be positive");
  }
  ValidateThreshold("video.person_detector.confidence_threshold",
                    config.video.person_detector_conf_threshold);
  ValidateThreshold("video.person_detector.nms_threshold",
                    config.video.person_detector_nms_threshold);
  ValidateThreshold("video.tracker.high_confidence_threshold",
                    config.video.tracker_high_confidence_threshold);
  ValidateThreshold("video.tracker.low_confidence_threshold",
                    config.video.tracker_low_confidence_threshold);
  ValidateThreshold("video.tracker.iou_threshold", config.video.tracker_iou_threshold);
  ValidateThreshold("video.tracker.center_distance_threshold",
                    config.video.tracker_center_distance_threshold);
  ValidateThreshold("face.det_confidence_threshold", config.face.det_confidence_threshold);
  ValidateThreshold("face.identity.cosine_match_threshold",
                    config.face.cosine_match_threshold);
  ValidateThreshold("vad.threshold", config.vad.threshold);
  ValidateThreshold("asd.speaking_threshold", config.asd.speaking_threshold);
  ValidateThreshold("fusion.visible_speaker_threshold",
                    config.fusion.visible_speaker_threshold);
  ValidateThreshold("fusion.offscreen_threshold", config.fusion.offscreen_threshold);
  if (config.fusion.offscreen_threshold > config.fusion.visible_speaker_threshold) {
    throw std::runtime_error(
        "fusion.offscreen_threshold must not exceed visible_speaker_threshold");
  }
}

}  // namespace

AppConfig LoadConfig(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& error) {
    throw std::runtime_error("failed to parse config " + path + ": " + error.what());
  }
  ValidateTopLevel(root);

  AppConfig config;
  const auto app = Section(root, "app");
  config.name = Read(app, "name", config.name);
  config.profile = Read(app, "profile", config.profile);

  const auto runtime = Section(root, "runtime");
  config.runtime.target_latency_ms =
      Read(runtime, "target_latency_ms", config.runtime.target_latency_ms);
  config.runtime.frame_queue_size =
      Read(runtime, "frame_queue_size", config.runtime.frame_queue_size);
  config.runtime.audio_queue_ms =
      Read(runtime, "audio_queue_ms", config.runtime.audio_queue_ms);
  config.runtime.worker_threads =
      Read(runtime, "worker_threads", config.runtime.worker_threads);
  config.runtime.use_cuda = Read(runtime, "use_cuda", config.runtime.use_cuda);
  config.runtime.use_tensorrt =
      Read(runtime, "use_tensorrt", config.runtime.use_tensorrt);
  config.runtime.allow_mock_inputs =
      Read(runtime, "allow_mock_inputs", config.runtime.allow_mock_inputs);
  config.runtime.session_log_path =
      Read(runtime, "session_log_path", config.runtime.session_log_path);
  config.runtime.event_ring_size =
      Read(runtime, "event_ring_size", config.runtime.event_ring_size);

  if (const auto capture = root["capture"]) {
    config.capture.backend = Read(capture, "backend", config.capture.backend);
  }
  if (const auto ui = root["ui"]) {
    if (const auto preview = ui["preview"]) {
      config.preview.width = Read(preview, "width", config.preview.width);
      config.preview.height = Read(preview, "height", config.preview.height);
      config.preview.fps = Read(preview, "fps", config.preview.fps);
      config.preview.jpeg_quality =
          Read(preview, "jpeg_quality", config.preview.jpeg_quality);
      config.preview.max_frame_age_ms =
          Read(preview, "max_frame_age_ms", config.preview.max_frame_age_ms);
    }
  }

  const auto video = Section(root, "video");
  config.video.source = Read(video, "source", config.video.source);
  config.video.camera_backend =
      Read(video, "camera_backend", config.video.camera_backend);
  config.video.camera_index = Read(video, "camera_index", config.video.camera_index);
  config.video.camera_builtin_only =
      Read(video, "camera_builtin_only", config.video.camera_builtin_only);
  config.video.render_bbox = Read(video, "render_bbox", config.video.render_bbox);
  config.video.camera_resolve_timeout_s =
      Read(video, "camera_resolve_timeout_s", config.video.camera_resolve_timeout_s);
  config.video.camera_name_allowlist =
      Read(video, "camera_name_allowlist", config.video.camera_name_allowlist);
  config.video.camera_name_denylist =
      Read(video, "camera_name_denylist", config.video.camera_name_denylist);
  config.video.width = Read(video, "width", config.video.width);
  config.video.height = Read(video, "height", config.video.height);
  config.video.fps = Read(video, "fps", config.video.fps);
  config.video.max_detection_result_age_ms =
      Read(video, "max_detection_result_age_ms",
           config.video.max_detection_result_age_ms);
  if (const auto detector = video["person_detector"]) {
    config.video.person_detector_backend =
        Read(detector, "backend", config.video.person_detector_backend);
    config.video.person_detector_model_path =
        Read(detector, "model_path", config.video.person_detector_model_path);
    config.video.person_detector_conf_threshold =
        Read(detector, "confidence_threshold", config.video.person_detector_conf_threshold);
    config.video.person_detector_imgsz =
        Read(detector, "imgsz", config.video.person_detector_imgsz);
    config.video.person_detector_nms_threshold =
        Read(detector, "nms_threshold", config.video.person_detector_nms_threshold);
    config.video.person_detector_detect_every_n_frames =
        Read(detector, "detect_every_n_frames",
             config.video.person_detector_detect_every_n_frames);
  }
  if (const auto tracker = video["tracker"]) {
    config.video.tracker_high_confidence_threshold =
        Read(tracker, "high_confidence_threshold",
             config.video.tracker_high_confidence_threshold);
    config.video.tracker_low_confidence_threshold =
        Read(tracker, "low_confidence_threshold",
             config.video.tracker_low_confidence_threshold);
    config.video.tracker_iou_threshold =
        Read(tracker, "iou_threshold", config.video.tracker_iou_threshold);
    config.video.tracker_center_distance_threshold =
        Read(tracker, "center_distance_threshold",
             config.video.tracker_center_distance_threshold);
    config.video.tracker_render_grace_ms =
        Read(tracker, "render_grace_ms", config.video.tracker_render_grace_ms);
    config.video.tracker_max_age_frames =
        Read(tracker, "max_age_frames", config.video.tracker_max_age_frames);
    config.video.tracker_min_hits =
        Read(tracker, "min_hits", config.video.tracker_min_hits);
  }

  const auto face = Section(root, "face");
  config.face.backend = Read(face, "backend", config.face.backend);
  config.face.model_pack = Read(face, "model_pack", config.face.model_pack);
  config.face.det_size = Read(face, "det_size", config.face.det_size);
  config.face.run_every_n_frames =
      Read(face, "run_every_n_frames", config.face.run_every_n_frames);
  config.face.det_confidence_threshold =
      Read(face, "det_confidence_threshold", config.face.det_confidence_threshold);
  config.face.hold_ms = Read(face, "hold_ms", config.face.hold_ms);
  config.face.asd_crop_ttl_ms =
      Read(face, "asd_crop_ttl_ms", config.face.asd_crop_ttl_ms);
  config.face.tracked_asd_crop_ttl_ms =
      Read(face, "tracked_asd_crop_ttl_ms", config.face.tracked_asd_crop_ttl_ms);
  config.face.identity_embedding_interval_ms =
      Read(face, "identity_embedding_interval_ms",
           config.face.identity_embedding_interval_ms);
  config.face.cosine_match_threshold =
      Read(face, "cosine_match_threshold", config.face.cosine_match_threshold);
  config.face.min_bbox_size_px =
      Read(face, "min_bbox_size_px", config.face.min_bbox_size_px);
  config.face.render_green_hold_ms =
      Read(face, "render_green_hold_ms", config.face.render_green_hold_ms);
  config.face.render_reacquire_hold_ms =
      Read(face, "render_reacquire_hold_ms", config.face.render_reacquire_hold_ms);
  config.face.asd_crop_max_geometry_age_ms =
      Read(face, "asd_crop_max_geometry_age_ms",
           config.face.asd_crop_max_geometry_age_ms);
  config.face.asd_crop_edge_margin_px =
      Read(face, "asd_crop_edge_margin_px", config.face.asd_crop_edge_margin_px);
  config.face.asd_crop_min_visible_ratio =
      Read(face, "asd_crop_min_visible_ratio", config.face.asd_crop_min_visible_ratio);
  if (const auto identity = face["identity"]) {
    config.face.cosine_match_threshold =
        Read(identity, "cosine_match_threshold", config.face.cosine_match_threshold);
  }

  const auto audio = Section(root, "audio");
  config.audio.source = Read(audio, "source", config.audio.source);
  config.audio.sample_rate = Read(audio, "sample_rate", config.audio.sample_rate);
  config.audio.channels = Read(audio, "channels", config.audio.channels);
  config.audio.chunk_ms = Read(audio, "chunk_ms", config.audio.chunk_ms);
  config.audio.builtin_only = Read(audio, "builtin_only", config.audio.builtin_only);
  config.audio.device = Read(audio, "device", config.audio.device);

  const auto vad = Section(root, "vad");
  config.vad.backend = Read(vad, "backend", config.vad.backend);
  config.vad.model_path = Read(vad, "model_path", config.vad.model_path);
  config.vad.threshold = Read(vad, "threshold", config.vad.threshold);
  config.vad.preroll_ms = Read(vad, "preroll_ms", config.vad.preroll_ms);
  config.vad.min_speech_ms = Read(vad, "min_speech_ms", config.vad.min_speech_ms);
  config.vad.min_silence_ms = Read(vad, "min_silence_ms", config.vad.min_silence_ms);
  config.vad.max_utterance_ms =
      Read(vad, "max_utterance_ms", config.vad.max_utterance_ms);

  const auto asr = Section(root, "asr");
  config.asr.backend = Read(asr, "backend", config.asr.backend);
  config.asr.model_dir = Read(asr, "model_dir", config.asr.model_dir);
  config.asr.streaming_backend =
      Read(asr, "streaming_backend", config.asr.streaming_backend);
  config.asr.streaming_model_dir =
      Read(asr, "streaming_model_dir", config.asr.streaming_model_dir);
  config.asr.language = Read(asr, "language", config.asr.language);
  config.asr.min_audio_ms = Read(asr, "min_audio_ms", config.asr.min_audio_ms);
  config.asr.silence_peak = Read(asr, "silence_peak", config.asr.silence_peak);
  config.asr.tail_padding_ms =
      Read(asr, "tail_padding_ms", config.asr.tail_padding_ms);
  config.asr.tail_silence_peak =
      Read(asr, "tail_silence_peak", config.asr.tail_silence_peak);
  config.asr.min_text_chars = Read(asr, "min_text_chars", config.asr.min_text_chars);
  config.asr.partial_result = Read(asr, "partial_result", config.asr.partial_result);
  config.asr.partial_min_audio_ms =
      Read(asr, "partial_min_audio_ms", config.asr.partial_min_audio_ms);
  config.asr.partial_min_text_chars =
      Read(asr, "partial_min_text_chars", config.asr.partial_min_text_chars);
  config.asr.partial_interval_ms =
      Read(asr, "partial_interval_ms", config.asr.partial_interval_ms);

  const auto asd = Section(root, "asd");
  config.asd.backend = Read(asd, "backend", config.asd.backend);
  config.asd.model_path = Read(asd, "model_path", config.asd.model_path);
  config.asd.reference_checkpoint_path =
      Read(asd, "reference_checkpoint_path", config.asd.reference_checkpoint_path);
  config.asd.window_ms = Read(asd, "window_ms", config.asd.window_ms);
  config.asd.hop_ms = Read(asd, "hop_ms", config.asd.hop_ms);
  config.asd.vad_hangover_ms =
      Read(asd, "vad_hangover_ms", config.asd.vad_hangover_ms);
  config.asd.min_crop_count =
      Read(asd, "min_crop_count", config.asd.min_crop_count);
  config.asd.max_crop_gap_ms =
      Read(asd, "max_crop_gap_ms", config.asd.max_crop_gap_ms);
  config.asd.max_candidate_tracks =
      Read(asd, "max_candidate_tracks", config.asd.max_candidate_tracks);
  config.asd.speaking_threshold =
      Read(asd, "speaking_threshold", config.asd.speaking_threshold);

  const auto diarization = Section(root, "diarization");
  config.diarization.segmentation_model =
      Read(diarization, "segmentation_model", config.diarization.segmentation_model);
  config.diarization.segmentation_threshold =
      Read(diarization, "segmentation_threshold",
           config.diarization.segmentation_threshold);
  config.diarization.segmentation_min_segment_ms =
      Read(diarization, "segmentation_min_segment_ms",
           config.diarization.segmentation_min_segment_ms);
  config.diarization.threshold =
      Read(diarization, "threshold", config.diarization.threshold);
  config.diarization.create_threshold =
      Read(diarization, "create_threshold", config.diarization.create_threshold);
  config.diarization.merge_threshold =
      Read(diarization, "merge_threshold", config.diarization.merge_threshold);
  config.diarization.model_path =
      Read(diarization, "model_path", config.diarization.model_path);

  const auto sync = Section(root, "sync");
  config.sync.video_time_offset_ms =
      Read(sync, "video_time_offset_ms", config.sync.video_time_offset_ms);
  config.sync.max_watermark_lag_ms =
      Read(sync, "max_watermark_lag_ms", config.sync.max_watermark_lag_ms);
  config.sync.reanchor_threshold_ms =
      Read(sync, "reanchor_threshold_ms", config.sync.reanchor_threshold_ms);
  config.sync.pll_alpha = Read(sync, "pll_alpha", config.sync.pll_alpha);
  config.sync.max_correction_per_chunk_ms =
      Read(sync, "max_correction_per_chunk_ms",
           config.sync.max_correction_per_chunk_ms);

  if (const auto diagnostics = root["diagnostics"]) {
    config.diagnostics.session_replay_enabled =
        Read(diagnostics, "session_replay_enabled",
             config.diagnostics.session_replay_enabled);
    config.diagnostics.session_replay_dir =
        Read(diagnostics, "session_replay_dir",
             config.diagnostics.session_replay_dir);
    config.diagnostics.session_replay_ttl_minutes =
        Read(diagnostics, "session_replay_ttl_minutes",
             config.diagnostics.session_replay_ttl_minutes);
  }

  const auto fusion = Section(root, "fusion");
  config.fusion.offscreen_threshold =
      Read(fusion, "offscreen_threshold", config.fusion.offscreen_threshold);
  config.fusion.visible_speaker_threshold =
      Read(fusion, "visible_speaker_threshold",
           config.fusion.visible_speaker_threshold);
  config.fusion.ambiguity_margin =
      Read(fusion, "ambiguity_margin", config.fusion.ambiguity_margin);
  config.fusion.allow_multi_speaker =
      Read(fusion, "allow_multi_speaker", config.fusion.allow_multi_speaker);

  const auto timeline = Section(root, "timeline");
  config.timeline.history_s = Read(timeline, "history_s", config.timeline.history_s);
  config.timeline.sample_interval_ms =
      Read(timeline, "sample_interval_ms", config.timeline.sample_interval_ms);
  config.timeline.active_threshold =
      Read(timeline, "active_threshold", config.timeline.active_threshold);
  config.timeline.visible_threshold =
      Read(timeline, "visible_threshold", config.timeline.visible_threshold);
  config.timeline.overlap_margin =
      Read(timeline, "overlap_margin", config.timeline.overlap_margin);
  config.timeline.interrupt_window_ms =
      Read(timeline, "interrupt_window_ms", config.timeline.interrupt_window_ms);

  if (const auto services = root["services"]) {
    if (!services.IsMap()) {
      throw std::runtime_error("services config section must be a mapping");
    }
    for (const auto& item : services) {
      const auto name = item.first.as<std::string>();
      const auto service = item.second;
      if (!service.IsMap()) {
        throw std::runtime_error("service config must be a mapping: " + name);
      }
      ServiceConfig service_config;
      service_config.enabled = Read(service, "enabled", service_config.enabled);
      service_config.host = Read(service, "host", service_config.host);
      service_config.port = Read(service, "port", service_config.port);
      config.services[name] = service_config;
    }
  }

  Validate(config);
  return config;
}

}  // namespace speaker_id
