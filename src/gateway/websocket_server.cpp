#include "speaker_id/core/websocket_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/time.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>

namespace speaker_id {

namespace {

// Self-contained SHA-1 implementation
void sha1(const std::string& input, uint8_t hash[20]) {
  uint32_t h0 = 0x67452301;
  uint32_t h1 = 0xEFCDAB89;
  uint32_t h2 = 0x98BADCFE;
  uint32_t h3 = 0x10325476;
  uint32_t h4 = 0xC3D2E1F0;

  std::vector<uint8_t> block(input.begin(), input.end());
  uint64_t bit_len = block.size() * 8;
  block.push_back(0x80);
  while ((block.size() + 8) % 64 != 0) {
    block.push_back(0x00);
  }
  for (int i = 7; i >= 0; --i) {
    block.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFF));
  }

  for (size_t chunk = 0; chunk < block.size(); chunk += 64) {
    uint32_t w[80] = {0};
    for (int i = 0; i < 16; ++i) {
      w[i] = (block[chunk + i * 4] << 24) |
             (block[chunk + i * 4 + 1] << 16) |
             (block[chunk + i * 4 + 2] << 8) |
             (block[chunk + i * 4 + 3]);
    }
    for (int i = 16; i < 80; ++i) {
      uint32_t val = w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16];
      w[i] = (val << 1) | (val >> 31);
    }

    uint32_t a = h0;
    uint32_t b = h1;
    uint32_t c = h2;
    uint32_t d = h3;
    uint32_t e = h4;

    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20) {
        f = (b & c) | ((~b) & d);
        k = 0x5A827999;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDC;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6;
      }
      uint32_t temp = ((a << 5) | (a >> 27)) + f + e + k + w[i];
      e = d;
      d = c;
      c = (b << 30) | (b >> 2);
      b = a;
      a = temp;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
  }

  hash[0] = h0 >> 24; hash[1] = h0 >> 16; hash[2] = h0 >> 8; hash[3] = h0;
  hash[4] = h1 >> 24; hash[5] = h1 >> 16; hash[6] = h1 >> 8; hash[7] = h1;
  hash[8] = h2 >> 24; hash[9] = h2 >> 16; hash[10] = h2 >> 8; hash[11] = h2;
  hash[12] = h3 >> 24; hash[13] = h3 >> 16; hash[14] = h3 >> 8; hash[15] = h3;
  hash[16] = h4 >> 24; hash[17] = h4 >> 16; hash[18] = h4 >> 8; hash[19] = h4;
}

// Self-contained Base64 encoder
std::string base64_encode(const uint8_t* data, size_t input_length) {
  const char encoding_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve(((input_length + 2) / 3) * 4);

  for (size_t i = 0; i < input_length; ) {
    uint32_t octet_a = i < input_length ? data[i++] : 0;
    uint32_t octet_b = i < input_length ? data[i++] : 0;
    uint32_t octet_c = i < input_length ? data[i++] : 0;

    uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

    output.push_back(encoding_table[(triple >> 3 * 6) & 0x3F]);
    output.push_back(encoding_table[(triple >> 2 * 6) & 0x3F]);
    output.push_back(encoding_table[(triple >> 1 * 6) & 0x3F]);
    output.push_back(encoding_table[(triple >> 0 * 6) & 0x3F]);
  }

  size_t mod = input_length % 3;
  if (mod == 1) {
    output[output.size() - 1] = '=';
    output[output.size() - 2] = '=';
  } else if (mod == 2) {
    output[output.size() - 1] = '=';
  }

  return output;
}

std::string Trim(std::string str) {
  str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), str.end());
  return str;
}

} // namespace

WebSocketServer::WebSocketServer(const std::string& host, int port)
    : host_(host), port_(port) {}

WebSocketServer::~WebSocketServer() {
  Stop();
}

bool WebSocketServer::Start() {
  if (running_.exchange(true)) {
    return false;
  }

  server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    running_ = false;
    return false;
  }

  int opt = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);

  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(server_fd_);
    server_fd_ = -1;
    running_ = false;
    return false;
  }

  if (listen(server_fd_, 16) < 0) {
    close(server_fd_);
    server_fd_ = -1;
    running_ = false;
    return false;
  }

  listen_thread_ = std::thread(&WebSocketServer::ListenLoop, this);
  return true;
}

void WebSocketServer::Stop() {
  if (!running_.exchange(false)) {
    return;
  }

  if (server_fd_ != -1) {
    shutdown(server_fd_, SHUT_RDWR);
    close(server_fd_);
    server_fd_ = -1;
  }

  if (listen_thread_.joinable()) {
    listen_thread_.join();
  }

  std::lock_guard<std::mutex> lock(clients_mu_);
  for (int fd : client_fds_) {
    close(fd);
  }
  client_fds_.clear();
}

void WebSocketServer::ListenLoop() {
  while (running_) {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (running_) {
        std::cerr << "WebSocket accept error\n";
      }
      break;
    }

    std::thread threadClient(&WebSocketServer::HandleClient, this, client_fd);
    threadClient.detach();
  }
}

void WebSocketServer::HandleClient(int client_fd) {
  // Read request
  std::vector<char> buffer(4096, 0);
  ssize_t bytes_read = read(client_fd, buffer.data(), buffer.size() - 1);
  if (bytes_read <= 0) {
    close(client_fd);
    return;
  }

  std::string request(buffer.data());
  std::istringstream stream(request);
  std::string line;
  std::string ws_key = "";

  while (std::getline(stream, line)) {
    if (line.back() == '\r') {
      line.pop_back();
    }
    std::string lower_line = line;
    std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower_line.find("sec-websocket-key:") == 0) {
      ws_key = Trim(line.substr(18));
    }
  }

  if (ws_key.empty()) {
    // Bad request
    std::string bad_response = "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n";
    (void)write(client_fd, bad_response.data(), bad_response.size());
    close(client_fd);
    return;
  }

  // Handshake response
  std::string magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  std::string accept_input = ws_key + magic;
  uint8_t hash[20] = {0};
  sha1(accept_input, hash);
  std::string accept_key = base64_encode(hash, 20);

  std::ostringstream response;
  response << "HTTP/1.1 101 Switching Protocols\r\n"
           << "Upgrade: websocket\r\n"
           << "Connection: Upgrade\r\n"
           << "Sec-WebSocket-Accept: " << accept_key << "\r\n\r\n";

  std::string response_str = response.str();
  if (write(client_fd, response_str.data(), response_str.size()) < 0) {
    close(client_fd);
    return;
  }

  // Set send timeout (e.g., 8ms) to prevent slow/blocked network clients from stalling the main video thread
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 8000;
  setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
  int no_sigpipe = 1;
  setsockopt(client_fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif

  {
    std::lock_guard<std::mutex> lock(clients_mu_);
    client_fds_.push_back(client_fd);
  }

  // Keep connection open by reading (draining client ping/pongs)
  std::vector<uint8_t> frame(1024);
  while (running_) {
    ssize_t n = read(client_fd, frame.data(), frame.size());
    if (n <= 0) {
      break; // connection closed
    }
  }

  {
    std::lock_guard<std::mutex> lock(clients_mu_);
    auto it = std::find(client_fds_.begin(), client_fds_.end(), client_fd);
    if (it != client_fds_.end()) {
      client_fds_.erase(it);
    }
  }
  close(client_fd);
}

void WebSocketServer::BroadcastBinary(const std::vector<uint8_t>& data) {
  if (data.empty()) return;

  // Frame binary payload
  std::vector<uint8_t> frame;
  frame.push_back(0x82); // FIN=1, Opcode=2 (Binary)

  size_t len = data.size();
  if (len < 126) {
    frame.push_back(static_cast<uint8_t>(len));
  } else if (len < 65536) {
    frame.push_back(126);
    frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
    frame.push_back(static_cast<uint8_t>(len & 0xFF));
  } else {
    frame.push_back(127);
    for (int i = 7; i >= 0; --i) {
      frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xFF));
    }
  }

  frame.insert(frame.end(), data.begin(), data.end());

  std::lock_guard<std::mutex> lock(clients_mu_);
  for (auto it = client_fds_.begin(); it != client_fds_.end(); ) {
    int fd = *it;
    int flags = MSG_DONTWAIT;
#ifdef MSG_NOSIGNAL
    flags |= MSG_NOSIGNAL;
#endif
    const auto written = send(fd, frame.data(), frame.size(), flags);
    if (written != static_cast<ssize_t>(frame.size())) {
      ++dropped_frames_;
      close(fd);
      it = client_fds_.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace speaker_id
