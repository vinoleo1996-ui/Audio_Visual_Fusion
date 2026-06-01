#pragma once

#include <map>
#include <string>
#include <vector>

namespace speaker_id {

struct ServiceConfig {
  bool enabled = false;
  std::string host = "127.0.0.1";
  int port = 0;
};

struct RuntimeConfig {
  int target_latency_ms = 500;
  int frame_queue_size = 6;
  int audio_queue_ms = 800;
  int worker_threads = 4;
  bool use_cuda = false;
  bool use_tensorrt = false;
  bool allow_mock_inputs = false;
  std::string session_log_path;
  std::size_t event_ring_size = 512;
};

struct CaptureConfig {
  std::string backend = "ffmpeg_pipe_pll";
};

struct PreviewConfig {
  int width = 960;
  int height = 540;
  int fps = 15;
  int jpeg_quality = 75;
  int max_frame_age_ms = 200;
};

struct VideoConfig {
  std::string source = "camera";
  std::string camera_backend = "opencv";
  int camera_index = 0;
  bool camera_builtin_only = true;
  bool render_bbox = false;
  float camera_resolve_timeout_s = 8.0F;
  int width = 1280;
  int height = 720;
  int fps = 30;
  std::string person_detector_backend = "ultralytics_yolo";
  std::string person_detector_model_path = "models/yolo/yolov8n.pt";
  float person_detector_conf_threshold = 0.35F;
  float person_detector_nms_threshold = 0.45F;
  int person_detector_imgsz = 640;
  int person_detector_detect_every_n_frames = 3;
  int max_detection_result_age_ms = 250;
  std::vector<std::string> camera_name_allowlist;
  std::vector<std::string> camera_name_denylist;
  float tracker_high_confidence_threshold = 0.45F;
  float tracker_low_confidence_threshold = 0.15F;
  float tracker_iou_threshold = 0.25F;
  float tracker_center_distance_threshold = 0.65F;
  int tracker_render_grace_ms = 350;
  int tracker_max_age_frames = 45;
  int tracker_min_hits = 3;
};

struct FaceConfig {
  std::string backend = "insightface";
  std::string model_pack = "buffalo_l";
  int det_size = 320;
  int run_every_n_frames = 5;
  float det_confidence_threshold = 0.45F;
  int hold_ms = 600;
  int asd_crop_ttl_ms = 1200;
  int tracked_asd_crop_ttl_ms = 2600;
  int identity_embedding_interval_ms = 750;
  float cosine_match_threshold = 0.38F;
  int min_bbox_size_px = 8;
  int render_green_hold_ms = 650;
  int render_reacquire_hold_ms = 1600;
  int asd_crop_max_geometry_age_ms = 650;
  int asd_crop_edge_margin_px = 12;
  float asd_crop_min_visible_ratio = 0.90F;
};

struct AudioConfig {
  std::string source = "microphone";
  int sample_rate = 16000;
  int channels = 1;
  int chunk_ms = 40;
  bool builtin_only = true;
  std::string device = "";
};

struct VadConfig {
  std::string backend = "silero_vad_onnx";
  std::string model_path = "models/vad/silero_vad_v6.onnx";
  float threshold = 0.25F;
  int preroll_ms = 480;
  int min_speech_ms = 450;
  int min_silence_ms = 1200;
  int max_utterance_ms = 12000;
};

struct AsrConfig {
  std::string backend = "sherpa_onnx_sensevoice";
  std::string model_dir = "models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
  std::string streaming_backend = "sherpa_zipformer_ctc";
  std::string streaming_model_dir = "models/asr/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01";
  std::string language = "zh";
  int min_audio_ms = 160;
  float silence_peak = 0.00005F;
  int tail_padding_ms = 180;
  float tail_silence_peak = 0.002F;
  int min_text_chars = 1;
  bool partial_result = true;
  int partial_min_audio_ms = 480;
  int partial_min_text_chars = 1;
  int partial_interval_ms = 400;
};

struct AsdConfig {
  std::string backend = "lr_asd_onnx";
  std::string model_path = "models/asd/lr-asd/lr_asd_talkset.onnx";
  std::string reference_checkpoint_path = "models/asd/lr-asd/finetuning_TalkSet.model";
  int window_ms = 1000;
  int hop_ms = 320;
  int vad_hangover_ms = 500;
  int min_crop_count = 20;
  int max_crop_gap_ms = 120;
  int max_candidate_tracks = 4;
  float speaking_threshold = 0.55F;
};

struct DiarizationConfig {
  std::string segmentation_model = "models/diarization/pyannote-segmentation-3.0.onnx";
  float segmentation_threshold = 0.50F;
  int segmentation_min_segment_ms = 200;
  float threshold = 0.50F;
  float create_threshold = 0.68F;
  float merge_threshold = 0.30F;
  std::string model_path = "models/voiceprint/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
};

struct SyncConfig {
  float video_time_offset_ms = 0.0F;
  int max_watermark_lag_ms = 250;
  int reanchor_threshold_ms = 500;
  double pll_alpha = 0.08;
  int max_correction_per_chunk_ms = 8;
};

struct DiagnosticsConfig {
  bool session_replay_enabled = false;
  std::string session_replay_dir = "output/session_replay";
  int session_replay_ttl_minutes = 60;
};

struct FusionConfig {
  float offscreen_threshold = 0.25F;
  float visible_speaker_threshold = 0.35F;
  float ambiguity_margin = 0.10F;
  bool allow_multi_speaker = true;
};

struct TimelineConfig {
  float history_s = 12.0F;
  float sample_interval_ms = 150.0F;
  float active_threshold = 0.55F;
  float visible_threshold = 0.35F;
  float overlap_margin = 0.12F;
  int interrupt_window_ms = 1200;
};

struct AppConfig {
  std::string name = "video-speaker-id";
  std::string profile = "default";
  RuntimeConfig runtime;
  CaptureConfig capture;
  PreviewConfig preview;
  VideoConfig video;
  FaceConfig face;
  AudioConfig audio;
  VadConfig vad;
  AsrConfig asr;
  AsdConfig asd;
  DiarizationConfig diarization;
  SyncConfig sync;
  DiagnosticsConfig diagnostics;
  FusionConfig fusion;
  TimelineConfig timeline;
  std::map<std::string, ServiceConfig> services;
};

AppConfig LoadConfig(const std::string& path);

}  // namespace speaker_id
