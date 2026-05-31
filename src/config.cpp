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
      "app",      "runtime", "services", "video",  "face", "audio", "vad",
      "asr",      "asd",     "diarization", "sync", "fusion", "timeline", "ui",
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
  if (config.runtime.target_latency_ms <= 0 || config.runtime.worker_threads <= 0) {
    throw std::runtime_error("runtime latency and worker thread limits must be positive");
  }
  if (config.video.width <= 0 || config.video.height <= 0 || config.video.fps <= 0) {
    throw std::runtime_error("video dimensions and fps must be positive");
  }
  if (config.video.person_detector_detect_every_n_frames <= 0 ||
      config.video.tracker_max_age_frames <= 0 || config.video.tracker_min_hits <= 0) {
    throw std::runtime_error("video detector and tracker cadence must be positive");
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
      config.asd.hop_ms > config.asd.window_ms) {
    throw std::runtime_error("invalid ASD window policy");
  }
  ValidateThreshold("video.person_detector.confidence_threshold",
                    config.video.person_detector_conf_threshold);
  ValidateThreshold("video.tracker.high_confidence_threshold",
                    config.video.tracker_high_confidence_threshold);
  ValidateThreshold("video.tracker.low_confidence_threshold",
                    config.video.tracker_low_confidence_threshold);
  ValidateThreshold("video.tracker.iou_threshold", config.video.tracker_iou_threshold);
  ValidateThreshold("vad.threshold", config.vad.threshold);
  ValidateThreshold("asd.speaking_threshold", config.asd.speaking_threshold);
  ValidateThreshold("fusion.visible_speaker_threshold",
                    config.fusion.visible_speaker_threshold);
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
  config.video.width = Read(video, "width", config.video.width);
  config.video.height = Read(video, "height", config.video.height);
  config.video.fps = Read(video, "fps", config.video.fps);
  if (const auto detector = video["person_detector"]) {
    config.video.person_detector_backend =
        Read(detector, "backend", config.video.person_detector_backend);
    config.video.person_detector_model_path =
        Read(detector, "model_path", config.video.person_detector_model_path);
    config.video.person_detector_conf_threshold =
        Read(detector, "confidence_threshold", config.video.person_detector_conf_threshold);
    config.video.person_detector_imgsz =
        Read(detector, "imgsz", config.video.person_detector_imgsz);
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
  config.face.cosine_match_threshold =
      Read(face, "cosine_match_threshold", config.face.cosine_match_threshold);
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

  const auto fusion = Section(root, "fusion");
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
