#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace speaker_id {

class WebSocketServer {
 public:
  WebSocketServer(const std::string& host, int port);
  ~WebSocketServer();

  bool Start();
  void Stop();

  // Send binary data (e.g. JPEG image) to all connected WebSocket clients
  void BroadcastBinary(const std::vector<uint8_t>& data);
  std::uint64_t DroppedFrames() const { return dropped_frames_.load(); }

  int Port() const { return port_; }

 private:
  void ListenLoop();
  void HandleClient(int client_fd);

  std::string host_;
  int port_;
  int server_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread listen_thread_;

  std::mutex clients_mu_;
  std::vector<int> client_fds_;
  std::atomic<std::uint64_t> dropped_frames_{0};
};

}  // namespace speaker_id
