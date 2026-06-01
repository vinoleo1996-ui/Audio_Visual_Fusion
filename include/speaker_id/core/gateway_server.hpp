#pragma once

#include "speaker_id/core/config.hpp"
#include "speaker_id/core/pipeline.hpp"
#include "speaker_id/core/websocket_server.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <map>

namespace httplib {
class Server;
}

namespace speaker_id {

class GatewayServer {
 public:
  GatewayServer(AppConfig config, std::shared_ptr<StreamingPipeline> pipeline);
  ~GatewayServer();

  bool Start();
  void Stop();

  void BroadcastVideoFrame(const std::vector<uint8_t>& jpeg_bytes,
                           std::uint64_t frame_sequence,
                           std::int64_t capture_timestamp_ms);
  std::uint64_t VideoFramesDropped() const;
  void ReportCameraStatus(
      bool ok,
      const std::string& error = "",
      const std::string& runtime_backend = "");
  void ReportAudioStatus(
      bool ok,
      const std::string& error = "",
      const std::string& runtime_backend = "");
  void ReportFatalError(const std::string& error);

  int Port() const { return config_.services.count("api") > 0 ? config_.services.at("api").port : 7050; }

 private:
  void HttpThreadLoop();
  void LoadFaceGallery();
  void SaveFaceGallery();
  void SaveFaceGalleryLocked();
  std::string SerializeFusionEvent(const FusionEvent& ev) const;

  AppConfig config_;
  std::shared_ptr<StreamingPipeline> pipeline_;
  std::unique_ptr<WebSocketServer> ws_server_;
  mutable std::mutex preview_mu_;
  std::condition_variable preview_cv_;
  std::vector<std::uint8_t> latest_preview_jpeg_;
  std::uint64_t latest_preview_sequence_ = 0;
  
  std::atomic<bool> running_{false};
  std::thread http_thread_;
  std::unique_ptr<httplib::Server> http_server_;

  mutable std::mutex status_mu_;
  bool camera_ok_ = false;
  bool audio_ok_ = false;
  std::string camera_runtime_backend_;
  std::string audio_runtime_backend_;
  std::string latest_error_;
  std::string fatal_error_;

  mutable std::mutex gallery_mu_;
  struct GalleryItem {
    std::string name;
    std::string background;
    std::vector<float> embedding;
  };
  std::map<int, GalleryItem> face_gallery_;
  std::string face_gallery_path_;
};

} // namespace speaker_id
