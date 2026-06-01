#include "speaker_id/core/config.hpp"
#include "speaker_id/core/audio_clock.hpp"
#include "speaker_id/core/pipeline.hpp"
#include "speaker_id/core/gateway_server.hpp"
#include "speaker_id/modules/asd.hpp"
#include "speaker_id/modules/asr.hpp"
#include "speaker_id/modules/diarization.hpp"
#include "speaker_id/modules/fusion.hpp"
#include "speaker_id/modules/vad.hpp"
#include "speaker_id/modules/vision.hpp"
#ifdef SPEAKER_ID_HAS_LIBAV
#include "speaker_id/capture/libav_video_capture.hpp"
#endif

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <csignal>
#include <cmath>
#include <cerrno>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

std::atomic<bool> g_running{true};

void SignalHandler(int) {
  g_running = false;
}

int64_t MonotonicNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Helper to resolve camera name dynamically on macOS
std::string ResolveCameraName(const speaker_id::AppConfig& config) {
  std::string cmd = "ffmpeg -f avfoundation -list_devices true -i \"\" 2>&1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return "0";
  }
  char buffer[256];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);

  const auto allowlist = config.video.camera_name_allowlist.empty()
                             ? std::vector<std::string>{"FaceTime", "Built-in", "builtin", "内置", "内建"}
                             : config.video.camera_name_allowlist;
  const auto denylist = config.video.camera_name_denylist.empty()
                            ? std::vector<std::string>{"iPhone", "iPad", "Continuity", "Desk View", "桌上视角", "Capture screen"}
                            : config.video.camera_name_denylist;

  size_t pos = 0;
  while ((pos = result.find("[AVFoundation indev @", pos)) != std::string::npos) {
    size_t line_end = result.find('\n', pos);
    if (line_end == std::string::npos) break;
    std::string line = result.substr(pos, line_end - pos);
    pos = line_end + 1;

    size_t brace_close = line.find(']', 20); // skip first block
    if (brace_close == std::string::npos) continue;

    size_t dev_idx_start = line.find('[', brace_close);
    if (dev_idx_start == std::string::npos) continue;
    size_t dev_idx_end = line.find(']', dev_idx_start);
    if (dev_idx_end == std::string::npos) continue;

    std::string name = line.substr(dev_idx_end + 1);
    // Trim
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    name.erase(name.find_last_not_of(" \t\r\n") + 1);

    if (name.empty()) continue;

    bool allowed = false;
    for (const auto& term : allowlist) {
      if (name.find(term) != std::string::npos) {
        allowed = true;
        break;
      }
    }
    bool denied = false;
    for (const auto& term : denylist) {
      if (name.find(term) != std::string::npos) {
        denied = true;
        break;
      }
    }

    if (allowed && !denied) {
      std::cout << "Resolved built-in camera device: " << name << "\n";
      return name;
    }
  }

  return "0";
}

// Helper to resolve audio name dynamically on macOS
std::string ResolveAudioName(const speaker_id::AppConfig& config) {
  if (!config.audio.device.empty()) {
    return config.audio.device;
  }

  std::string cmd = "ffmpeg -f avfoundation -list_devices true -i \"\" 2>&1";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return "default";
  }
  char buffer[256];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);

  size_t audio_sec = result.find("AVFoundation audio devices:");
  if (audio_sec == std::string::npos) {
    return "default";
  }

  // Allowlist & Denylist for built-in microphone
  std::vector<std::string> allowlist = {"Microphone", "Built-in", "built-in", "麦克风", "MacBook"};
  std::vector<std::string> denylist = {"iPhone", "iPad", "Continuity", "WeMeet", "Lark", "Zoom", "Virtual", "BlackHole", "Loopback"};

  size_t pos = audio_sec;
  while ((pos = result.find("[AVFoundation indev @", pos)) != std::string::npos) {
    size_t line_end = result.find('\n', pos);
    if (line_end == std::string::npos) break;
    std::string line = result.substr(pos, line_end - pos);
    pos = line_end + 1;

    size_t brace_close = line.find(']', 20); // skip first block
    if (brace_close == std::string::npos) continue;

    size_t dev_idx_start = line.find('[', brace_close);
    if (dev_idx_start == std::string::npos) continue;
    size_t dev_idx_end = line.find(']', dev_idx_start);
    if (dev_idx_end == std::string::npos) continue;

    std::string name = line.substr(dev_idx_end + 1);
    // Trim
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    name.erase(name.find_last_not_of(" \t\r\n") + 1);

    if (name.empty()) continue;

    if (config.audio.builtin_only) {
      bool allowed = false;
      for (const auto& term : allowlist) {
        if (name.find(term) != std::string::npos) {
          allowed = true;
          break;
        }
      }
      bool denied = false;
      for (const auto& term : denylist) {
        if (name.find(term) != std::string::npos) {
          denied = true;
          break;
        }
      }

      if (allowed && !denied) {
        std::cout << "Resolved built-in audio device: " << name << "\n";
        return name;
      }
    } else {
      std::cout << "Resolved default audio device: " << name << "\n";
      return name;
    }
  }

  return "default";
}

// Subprocess mic capture loop
void AudioCaptureLoop(
    const speaker_id::AppConfig& config,
    std::shared_ptr<speaker_id::StreamingPipeline> pipeline,
    std::shared_ptr<speaker_id::GatewayServer> gateway) {
  std::string cmd;
#ifdef __APPLE__
  std::string audio_device = ResolveAudioName(config);
  cmd = "ffmpeg -loglevel quiet -f avfoundation -i \":" + audio_device + "\" -ar " +
        std::to_string(config.audio.sample_rate) + " -ac " +
        std::to_string(config.audio.channels) + " -f s16le -";
#else
  std::string audio_device = config.audio.device.empty() ? "default" : config.audio.device;
  cmd = "ffmpeg -loglevel quiet -f alsa -i " + audio_device + " -ar " +
        std::to_string(config.audio.sample_rate) + " -ac " +
        std::to_string(config.audio.channels) + " -f s16le -";
#endif

  std::cout << "Starting audio capture subprocess: " << cmd << "\n";
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) {
    if (!config.runtime.allow_mock_inputs) {
      const std::string error =
          "failed to spawn audio capture subprocess and mock inputs are disabled";
      std::cerr << "Fatal: " << error << ".\n";
      gateway->ReportAudioStatus(false, error);
      gateway->ReportFatalError(error);
      g_running = false;
      return;
    }
    std::cerr << "Audio capture unavailable. Using explicitly enabled mock audio stream.\n";
    gateway->ReportAudioStatus(false, "explicit mock audio stream active", "mock");
    int64_t timestamp_ms = MonotonicNowMs();
    while (g_running) {
      std::this_thread::sleep_for(std::chrono::milliseconds(config.audio.chunk_ms));
      speaker_id::AudioChunkEvent chunk;
      chunk.stream_id = "mic";
      chunk.timestamp_ms = timestamp_ms;
      chunk.duration_ms = config.audio.chunk_ms;
      chunk.sample_rate = config.audio.sample_rate;
      chunk.data.assign(config.audio.sample_rate * config.audio.chunk_ms / 1000, 0.0f);
      pipeline->PushAudio(chunk);
      timestamp_ms += config.audio.chunk_ms;
    }
    return;
  }
  const int audio_fd = fileno(fp);
  fcntl(audio_fd, F_SETFL, fcntl(audio_fd, F_GETFL, 0) | O_NONBLOCK);

  const size_t chunk_samples =
      config.audio.sample_rate * config.audio.channels * config.audio.chunk_ms / 1000;
  std::vector<int16_t> read_buf(chunk_samples);
  size_t audio_bytes_offset = 0;
  int64_t frame_count = 0;
  speaker_id::AudioMonotonicPll audio_clock(
      config.sync.reanchor_threshold_ms, config.sync.pll_alpha,
      config.sync.max_correction_per_chunk_ms);
  bool audio_reported_ok = false;

  while (g_running) {
    const auto bytes_read = read(
        audio_fd,
        reinterpret_cast<char*>(read_buf.data()) + audio_bytes_offset,
        chunk_samples * sizeof(int16_t) - audio_bytes_offset);
    if (bytes_read < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ||
                           errno == EINTR)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    if (bytes_read <= 0) {
      if (!g_running) {
        break;
      }
      const std::string error = "audio stream reached EOF or returned an error";
      std::cerr << "Fatal: " << error << ".\n";
      gateway->ReportAudioStatus(false, error);
      gateway->ReportFatalError(error);
      g_running = false;
      break;
    }
    audio_bytes_offset += static_cast<std::size_t>(bytes_read);
    if (audio_bytes_offset < chunk_samples * sizeof(int16_t)) {
      continue;
    }
    const auto read_samples = chunk_samples;
    audio_bytes_offset = 0;
    if (!audio_reported_ok) {
#ifdef __APPLE__
      gateway->ReportAudioStatus(true, "", "ffmpeg_avfoundation");
#else
      gateway->ReportAudioStatus(true, "", "ffmpeg_alsa");
#endif
      audio_reported_ok = true;
    }

    speaker_id::AudioChunkEvent chunk;
    chunk.stream_id = "mic";
    chunk.duration_ms = static_cast<int>(
        read_samples * 1000 / (config.audio.sample_rate * config.audio.channels));
    const auto clock =
        audio_clock.Stamp(MonotonicNowMs(), chunk.duration_ms);
    chunk.timestamp_ms = clock.timestamp_ms;
    chunk.clock_drift_ms = clock.drift_ms;
    chunk.clock_drift_ms_per_min = clock.drift_ms_per_min;
    chunk.clock_reanchored = clock.reanchored;
    chunk.sample_rate = config.audio.sample_rate;
    chunk.data.reserve(read_samples);
    double sum_sq = 0.0;
    for (size_t i = 0; i < read_samples; ++i) {
      float val = static_cast<float>(read_buf[i]) / 32768.0f;
      chunk.data.push_back(val);
      sum_sq += val * val;
    }
    double rms = std::sqrt(sum_sq / std::max<size_t>(1, read_samples));
    static int chunk_count = 0;
    if (chunk_count++ % 100 == 0) {
      std::cout << "[AudioCapture] Chunk " << chunk_count << ", read " << read_samples << " samples, RMS: " << rms << "\n";
    }

    pipeline->PushAudio(chunk);
    frame_count++;
  }

  fclose(fp);
}

// Global thread-safe camera frame buffer to prevent pipe accumulation latency
std::mutex g_video_mu;
cv::Mat g_latest_video_frame;
int64_t g_latest_video_timestamp_ms = 0;
std::atomic<bool> g_new_video_frame{false};
std::atomic<std::uint64_t> g_pipe_frames_dropped{0};
std::atomic<std::size_t> g_pipe_backlog_bytes{0};
std::mutex g_preview_mu;
struct PreviewFrame {
  cv::Mat frame;
  std::uint64_t sequence = 0;
  std::int64_t capture_timestamp_ms = 0;
};
PreviewFrame g_latest_preview_frame;
std::atomic<bool> g_new_preview_frame{false};
std::atomic<std::uint64_t> g_preview_frames_dropped{0};

void FfmpegVideoCaptureLoop(FILE* fp, int width, int height) {
  size_t frame_bytes = width * height * 3;
  std::vector<uint8_t> buffer(frame_bytes);
  const int video_fd = fileno(fp);
  fcntl(video_fd, F_SETFL, fcntl(video_fd, F_GETFL, 0) | O_NONBLOCK);
  size_t offset = 0;
  while (g_running) {
    const auto bytes_read = read(video_fd, buffer.data() + offset, frame_bytes - offset);
    if (bytes_read < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ||
                           errno == EINTR)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if (bytes_read <= 0) {
      if (!g_running) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    offset += static_cast<std::size_t>(bytes_read);
    if (offset == frame_bytes) {
      cv::Mat frame(height, width, CV_8UC3, buffer.data());
      {
        std::lock_guard<std::mutex> lock(g_video_mu);
        g_latest_video_frame = frame.clone();
        g_latest_video_timestamp_ms = MonotonicNowMs();
        g_new_video_frame = true;
      }
      offset = 0;
      int available_bytes = 0;
      if (ioctl(video_fd, FIONREAD, &available_bytes) == 0) {
        g_pipe_backlog_bytes = static_cast<std::size_t>(std::max(0, available_bytes));
        while (available_bytes >= static_cast<int>(frame_bytes * 2U)) {
          std::size_t discarded = 0;
          while (discarded < frame_bytes) {
            const auto dropped = read(video_fd, buffer.data() + discarded,
                                      frame_bytes - discarded);
            if (dropped <= 0) {
              break;
            }
            discarded += static_cast<std::size_t>(dropped);
          }
          if (discarded != frame_bytes) {
            break;
          }
          ++g_pipe_frames_dropped;
          available_bytes -= static_cast<int>(frame_bytes);
        }
      }
    }
  }
}

void PreviewEncodeLoop(
    const speaker_id::PreviewConfig& config,
    const std::string& frame_age_source,
    std::shared_ptr<speaker_id::StreamingPipeline> pipeline,
    std::shared_ptr<speaker_id::GatewayServer> gateway) {
  const auto cadence = std::chrono::milliseconds(
      std::max(1, 1000 / std::max(1, config.fps)));
  auto next_encode_at = std::chrono::steady_clock::now();
  while (g_running) {
    const auto now = std::chrono::steady_clock::now();
    if (now < next_encode_at) {
      std::this_thread::sleep_until(next_encode_at);
    }
    next_encode_at = std::chrono::steady_clock::now() + cadence;
    PreviewFrame preview;
    {
      std::lock_guard<std::mutex> lock(g_preview_mu);
      if (g_new_preview_frame && !g_latest_preview_frame.frame.empty()) {
        preview = g_latest_preview_frame;
        g_new_preview_frame = false;
      }
    }
    if (preview.frame.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    try {
      const auto encode_started_at = std::chrono::steady_clock::now();
      cv::Mat resized;
      cv::resize(preview.frame, resized, cv::Size(config.width, config.height),
                 0.0, 0.0, cv::INTER_AREA);
      std::vector<uint8_t> jpeg_bytes;
      std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, config.jpeg_quality};
      cv::imencode(".jpg", resized, jpeg_bytes, params);
      gateway->BroadcastVideoFrame(
          jpeg_bytes, preview.sequence, preview.capture_timestamp_ms);
      const auto sent_at = std::chrono::steady_clock::now();
      const double encode_latency_ms =
          std::chrono::duration<double, std::milli>(sent_at - encode_started_at).count();
      const double capture_to_send_ms =
          static_cast<double>(MonotonicNowMs() - preview.capture_timestamp_ms);
      pipeline->ReportPreviewFrame(
          encode_latency_ms, capture_to_send_ms, g_preview_frames_dropped.load(),
          gateway->VideoFramesDropped(), g_pipe_frames_dropped.load(),
          g_pipe_backlog_bytes.load(), frame_age_source);
    } catch (const std::exception& error) {
      std::cerr << "Preview encode error: " << error.what() << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;
  std::string config_path = "configs/live_mac.yaml";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--config" && i + 1 < argc) {
      config_path = argv[i + 1];
    }
  }

  std::cout << "Loading configuration: " << config_path << "\n";
  speaker_id::AppConfig config;
  try {
    config = speaker_id::LoadConfig(config_path);
  } catch (const std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << "\n";
    return 1;
  }
#ifndef SPEAKER_ID_HAS_LIBAV
  if (config.capture.backend == "libav_pts") {
    std::cerr << "Fatal error: capture.backend=libav_pts was selected, but the "
                 "libav PTS adapter is not linked in this Mac build yet.\n";
    return 1;
  }
#endif
  if (config.runtime.use_cuda || config.runtime.use_tensorrt) {
    std::cerr
        << "Fatal error: this C++ build exposes CPU ONNX Runtime providers only. "
        << "The requested CUDA/TensorRT Orin profile requires linked and benchmarked "
        << "Jetson execution providers.\n";
    return 1;
  }

  // Register signal handler for clean shutdown
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  std::shared_ptr<speaker_id::YoloVisionBackend> vision;
  std::shared_ptr<speaker_id::SileroVadBackend> vad;
  std::shared_ptr<speaker_id::SherpaAsrBackend> asr;
  std::shared_ptr<speaker_id::OrtDiarizationBackend> diarization;
  std::shared_ptr<speaker_id::AsdBackend> asd;
  std::shared_ptr<speaker_id::RuleFusionBackend> fusion;
  try {
    std::cout << "Initializing Vision module: " << config.video.person_detector_model_path << "\n";
    vision = std::make_shared<speaker_id::YoloVisionBackend>(
        config.video.person_detector_model_path,
        std::min(config.video.person_detector_conf_threshold,
                 config.video.tracker_low_confidence_threshold),
        config.video.person_detector_imgsz, config.video.person_detector_nms_threshold);

    std::cout << "Initializing VAD module: " << config.vad.model_path << "\n";
    vad = std::make_shared<speaker_id::SileroVadBackend>(
        config.vad.model_path,
        config.vad.threshold,
        config.vad.preroll_ms,
        config.vad.min_silence_ms,
        config.vad.min_speech_ms,
        config.vad.max_utterance_ms);

    std::cout << "Initializing ASR module: " << config.asr.model_dir << "\n";
    asr = std::make_shared<speaker_id::SherpaAsrBackend>(
        config.asr, config.runtime.worker_threads);

    std::cout << "Initializing Diarization module: " << config.diarization.segmentation_model << "\n";
    diarization = std::make_shared<speaker_id::OrtDiarizationBackend>(
        config.diarization.segmentation_model,
        config.diarization.model_path,
        config.diarization.segmentation_threshold,
        config.diarization.segmentation_min_segment_ms);

    if (config.asd.backend != "lr_asd_onnx") {
      throw std::runtime_error("unsupported C++ ASD backend: " + config.asd.backend);
    }
    std::cout << "Initializing ASD module: " << config.asd.model_path << "\n";
    asd = std::make_shared<speaker_id::OrtLrAsdBackend>(config.asd.model_path);

    std::cout << "Initializing Fusion module...\n";
    fusion = std::make_shared<speaker_id::RuleFusionBackend>(
        config.video.width, config.asd.speaking_threshold,
        config.fusion.visible_speaker_threshold, config.fusion.offscreen_threshold,
        config.fusion.ambiguity_margin, config.fusion.allow_multi_speaker);
  } catch (const std::exception& error) {
    std::cerr << "Fatal error: runtime backend initialization failed: " << error.what()
              << "\n";
    return 1;
  }

  // Assemble pipeline
  auto pipeline = std::make_shared<speaker_id::StreamingPipeline>(
      config, vision, vad, asr, asd, fusion, diarization);

  // Create gateway server
  auto gateway = std::make_shared<speaker_id::GatewayServer>(config, pipeline);

  // Start modules
  pipeline->Start();
  gateway->Start();

  // Spawn audio capture thread
  std::thread audio_capture_thread(AudioCaptureLoop, config, pipeline, gateway);
  std::thread preview_encode_thread(
      PreviewEncodeLoop, config.preview,
      config.capture.backend == "libav_pts" ? "libav_pts" : "arrival_estimate",
      pipeline, gateway);

  // Main camera capture loop
  cv::VideoCapture cap;
  FILE* video_fp = nullptr;
  bool use_ffmpeg_video = false;
#ifdef SPEAKER_ID_HAS_LIBAV
  std::unique_ptr<speaker_id::LibavVideoCapture> libav_video;
  bool use_libav_video = false;
#endif

#ifdef __APPLE__
  if (config.video.source == "camera") {
    std::string camera_name = ResolveCameraName(config);
    if (camera_name != "0") {
#ifdef SPEAKER_ID_HAS_LIBAV
      if (config.capture.backend == "libav_pts") {
        libav_video = std::make_unique<speaker_id::LibavVideoCapture>();
        std::string error;
        if (libav_video->Open(camera_name, config.video.width, config.video.height,
                              config.video.fps, error)) {
          use_libav_video = true;
        } else {
          std::cerr << "Fatal error: " << error << "\n";
          gateway->ReportCameraStatus(false, error, "libav_pts");
          gateway->ReportFatalError(error);
          g_running = false;
        }
      } else
#endif
      {
        std::string ffmpeg_cmd = "ffmpeg -loglevel quiet -fflags nobuffer -flags low_delay -f avfoundation -r " + std::to_string(config.video.fps) +
                                 " -video_size " + std::to_string(config.video.width) + "x" + std::to_string(config.video.height) +
                                 " -i \"" + camera_name + "\" -pix_fmt bgr24 -f rawvideo -";
        std::cout << "Opening FaceTime camera via FFmpeg command: " << ffmpeg_cmd << "\n";
        video_fp = popen(ffmpeg_cmd.c_str(), "r");
        if (video_fp) {
          use_ffmpeg_video = true;
        } else {
          std::cerr << "Failed to open camera via FFmpeg. Falling back to OpenCV...\n";
        }
      }
    }
  }
#endif

  std::thread ffmpeg_video_thread;
  if (use_ffmpeg_video && video_fp) {
    ffmpeg_video_thread = std::thread(FfmpegVideoCaptureLoop, video_fp, config.video.width, config.video.height);
  }

  if (!use_ffmpeg_video
#ifdef SPEAKER_ID_HAS_LIBAV
      && !use_libav_video
#endif
      && config.video.source == "camera") {
    std::cout << "Opening camera index " << config.video.camera_index << "...\n";
    cap.open(config.video.camera_index);
    if (cap.isOpened()) {
      cap.set(cv::CAP_PROP_FRAME_WIDTH, config.video.width);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.video.height);
      cap.set(cv::CAP_PROP_FPS, config.video.fps);
      std::cout << "Camera resolution set to: " << config.video.width << "x" << config.video.height << "\n";
      gateway->ReportCameraStatus(true, "", "opencv");
    } else {
      if (config.runtime.allow_mock_inputs) {
        std::cerr << "Camera unavailable. Using explicitly enabled mock video stream.\n";
        gateway->ReportCameraStatus(false, "explicit mock video stream active");
      } else {
        const std::string error = "could not open camera device and mock inputs are disabled";
        std::cerr << "Fatal: " << error << ".\n";
        gateway->ReportCameraStatus(false, error);
        gateway->ReportFatalError(error);
        g_running = false;
      }
    }
  }

  int64_t frame_id = 0;
  int frame_interval_ms = 1000 / (config.video.fps > 0 ? config.video.fps : 30);

  while (g_running) {
    cv::Mat frame;
    int64_t timestamp_ms = 0;

    bool read_success = false;
#ifdef SPEAKER_ID_HAS_LIBAV
    if (use_libav_video && libav_video) {
      speaker_id::CapturedVideoFrame captured;
      std::string error;
      if (libav_video->ReadFrame(captured, error)) {
        frame = std::move(captured.bgr);
        timestamp_ms = captured.capture_timestamp_ms;
        gateway->ReportCameraStatus(true, "", "libav_pts");
        read_success = true;
      } else {
        gateway->ReportCameraStatus(false, error, "libav_pts");
        gateway->ReportFatalError(error);
        g_running = false;
        break;
      }
    } else
#endif
    if (use_ffmpeg_video && video_fp) {
      if (g_new_video_frame) {
        {
          std::lock_guard<std::mutex> lock(g_video_mu);
          frame = g_latest_video_frame.clone();
          timestamp_ms = g_latest_video_timestamp_ms;
          g_new_video_frame = false;
        }
        gateway->ReportCameraStatus(true, "", "ffmpeg_avfoundation");
        read_success = true;
      } else {
        // Sleep a tiny bit to prevent spinning, wait for the next frame
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        continue;
      }
    } else if (cap.isOpened()) {
      if (cap.read(frame)) {
        timestamp_ms = MonotonicNowMs();
        read_success = true;
      }
    }

    if (!read_success) {
      if (!config.runtime.allow_mock_inputs) {
        const std::string error = "video capture failed and mock inputs are disabled";
        std::cerr << "Fatal: " << error << ".\n";
        gateway->ReportCameraStatus(false, error);
        gateway->ReportFatalError(error);
        g_running = false;
        break;
      }
      timestamp_ms = MonotonicNowMs();
      frame = cv::Mat::zeros(config.video.height, config.video.width, CV_8UC3);
      cv::putText(frame, "C++ Mock Video Feed Active", cv::Point(50, 100), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 200, 100), 2);
      std::this_thread::sleep_for(std::chrono::milliseconds(frame_interval_ms));
    }

    if (!frame.empty()) {
      speaker_id::FrameEvent frame_event;
      frame_event.stream_id = "camera";
      frame_event.frame_id = frame_id++;
      frame_event.timestamp_ms = timestamp_ms;
      frame_event.width = frame.cols;
      frame_event.height = frame.rows;
      const cv::Mat contiguous_frame = frame.isContinuous() ? frame : frame.clone();
      frame_event.data.assign(
          contiguous_frame.datastart,
          contiguous_frame.dataend);
      pipeline->PushFrame(std::move(frame_event));

      {
        std::lock_guard<std::mutex> lock(g_preview_mu);
        if (g_new_preview_frame) {
          ++g_preview_frames_dropped;
        }
        g_latest_preview_frame =
            PreviewFrame{frame.clone(), static_cast<std::uint64_t>(frame_id - 1), timestamp_ms};
        g_new_preview_frame = true;
      }
    }
  }

  std::cout << "Shutting down C++ Gateway and Pipeline...\n";
  g_running = false;

  if (ffmpeg_video_thread.joinable()) {
    ffmpeg_video_thread.join();
  }

  if (audio_capture_thread.joinable()) {
    audio_capture_thread.join();
  }

  gateway->Stop();
  if (preview_encode_thread.joinable()) {
    preview_encode_thread.join();
  }
  pipeline->Stop();

  if (cap.isOpened()) {
    cap.release();
  }
#ifdef SPEAKER_ID_HAS_LIBAV
  if (libav_video) {
    libav_video->Close();
  }
#endif
  if (video_fp) {
    fclose(video_fp);
  }

  std::cout << "Shutdown complete.\n";
  return 0;
}
