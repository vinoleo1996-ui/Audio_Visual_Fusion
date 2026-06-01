#include "speaker_id/capture/libav_video_capture.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <chrono>
#include <sstream>

namespace speaker_id {
namespace {

std::int64_t MonotonicNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string AvError(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

}  // namespace

struct LibavVideoCapture::Impl {
  AVFormatContext* format = nullptr;
  AVCodecContext* decoder = nullptr;
  AVFrame* decoded = nullptr;
  AVPacket* packet = nullptr;
  SwsContext* scaler = nullptr;
  int video_stream = -1;
  AVRational time_base{1, 1000};
  std::int64_t first_pts = AV_NOPTS_VALUE;
  std::int64_t first_monotonic_ms = 0;
  std::uint64_t sequence = 0;
};

LibavVideoCapture::LibavVideoCapture() : impl_(std::make_unique<Impl>()) {}

LibavVideoCapture::~LibavVideoCapture() {
  Close();
}

bool LibavVideoCapture::Open(
    const std::string& device_name, int width, int height, int fps,
    std::string& error) {
  Close();
  avdevice_register_all();
  const AVInputFormat* input = av_find_input_format("avfoundation");
  if (input == nullptr) {
    error = "libav avfoundation input is unavailable";
    return false;
  }
  AVDictionary* options = nullptr;
  const auto size = std::to_string(width) + "x" + std::to_string(height);
  av_dict_set(&options, "video_size", size.c_str(), 0);
  av_dict_set(&options, "framerate", std::to_string(fps).c_str(), 0);
  av_dict_set(&options, "pixel_format", "uyvy422", 0);
  const int open_result =
      avformat_open_input(&impl_->format, device_name.c_str(), input, &options);
  av_dict_free(&options);
  if (open_result < 0) {
    error = "libav open input failed: " + AvError(open_result);
    Close();
    return false;
  }
  const int info_result = avformat_find_stream_info(impl_->format, nullptr);
  if (info_result < 0) {
    error = "libav stream info failed: " + AvError(info_result);
    Close();
    return false;
  }
  for (unsigned int index = 0; index < impl_->format->nb_streams; ++index) {
    if (impl_->format->streams[index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      impl_->video_stream = static_cast<int>(index);
      break;
    }
  }
  if (impl_->video_stream < 0) {
    error = "libav camera input has no video stream";
    Close();
    return false;
  }
  auto* stream = impl_->format->streams[impl_->video_stream];
  const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
  if (codec == nullptr) {
    error = "libav camera decoder is unavailable";
    Close();
    return false;
  }
  impl_->decoder = avcodec_alloc_context3(codec);
  if (impl_->decoder == nullptr ||
      avcodec_parameters_to_context(impl_->decoder, stream->codecpar) < 0 ||
      avcodec_open2(impl_->decoder, codec, nullptr) < 0) {
    error = "libav camera decoder initialization failed";
    Close();
    return false;
  }
  impl_->decoded = av_frame_alloc();
  impl_->packet = av_packet_alloc();
  impl_->time_base = stream->time_base;
  impl_->first_monotonic_ms = MonotonicNowMs();
  return impl_->decoded != nullptr && impl_->packet != nullptr;
}

bool LibavVideoCapture::ReadFrame(CapturedVideoFrame& frame, std::string& error) {
  if (impl_->format == nullptr || impl_->decoder == nullptr) {
    error = "libav camera is not open";
    return false;
  }
  while (true) {
    const int read_result = av_read_frame(impl_->format, impl_->packet);
    if (read_result < 0) {
      error = "libav read frame failed: " + AvError(read_result);
      return false;
    }
    if (impl_->packet->stream_index != impl_->video_stream) {
      av_packet_unref(impl_->packet);
      continue;
    }
    const int send_result = avcodec_send_packet(impl_->decoder, impl_->packet);
    av_packet_unref(impl_->packet);
    if (send_result < 0) {
      error = "libav send packet failed: " + AvError(send_result);
      return false;
    }
    const int receive_result = avcodec_receive_frame(impl_->decoder, impl_->decoded);
    if (receive_result == AVERROR(EAGAIN)) {
      continue;
    }
    if (receive_result < 0) {
      error = "libav receive frame failed: " + AvError(receive_result);
      return false;
    }

    cv::Mat bgr(impl_->decoded->height, impl_->decoded->width, CV_8UC3);
    impl_->scaler = sws_getCachedContext(
        impl_->scaler, impl_->decoded->width, impl_->decoded->height,
        static_cast<AVPixelFormat>(impl_->decoded->format),
        impl_->decoded->width, impl_->decoded->height, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (impl_->scaler == nullptr) {
      error = "libav pixel conversion initialization failed";
      return false;
    }
    std::uint8_t* output[] = {bgr.data, nullptr, nullptr, nullptr};
    int output_stride[] = {static_cast<int>(bgr.step), 0, 0, 0};
    sws_scale(impl_->scaler, impl_->decoded->data, impl_->decoded->linesize, 0,
              impl_->decoded->height, output, output_stride);

    const auto pts = impl_->decoded->best_effort_timestamp;
    if (impl_->first_pts == AV_NOPTS_VALUE && pts != AV_NOPTS_VALUE) {
      impl_->first_pts = pts;
      impl_->first_monotonic_ms = MonotonicNowMs();
    }
    frame.bgr = std::move(bgr);
    frame.sequence = ++impl_->sequence;
    frame.capture_timestamp_ms =
        pts == AV_NOPTS_VALUE || impl_->first_pts == AV_NOPTS_VALUE
            ? MonotonicNowMs()
            : impl_->first_monotonic_ms +
                  av_rescale_q(pts - impl_->first_pts, impl_->time_base,
                               AVRational{1, 1000});
    return true;
  }
}

void LibavVideoCapture::Close() {
  if (!impl_) {
    return;
  }
  sws_freeContext(impl_->scaler);
  impl_->scaler = nullptr;
  av_packet_free(&impl_->packet);
  av_frame_free(&impl_->decoded);
  avcodec_free_context(&impl_->decoder);
  if (impl_->format != nullptr) {
    avformat_close_input(&impl_->format);
  }
  impl_->video_stream = -1;
  impl_->first_pts = AV_NOPTS_VALUE;
  impl_->sequence = 0;
}

}  // namespace speaker_id
