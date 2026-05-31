#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <string>

namespace speaker_id {

class Module {
 public:
  virtual ~Module() = default;
  virtual std::string Name() const = 0;
  virtual bool Start() = 0;
  virtual void Stop() = 0;
};

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  bool Push(T value) {
    std::unique_lock<std::mutex> lock(mu_);
    not_full_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
    if (closed_) {
      return false;
    }
    queue_.push(std::move(value));
    not_empty_.notify_one();
    return true;
  }

  bool PushOrDrop(T value) {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      return false;
    }
    if (queue_.size() >= capacity_) {
      queue_.pop(); // Discard oldest frame to keep latency minimal
    }
    queue_.push(std::move(value));
    not_empty_.notify_one();
    return true;
  }

  std::optional<T> Pop() {
    std::unique_lock<std::mutex> lock(mu_);
    not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return std::nullopt;
    }
    T value = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return value;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  std::size_t capacity_;
  std::queue<T> queue_;
  std::mutex mu_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool closed_ = false;
};

}  // namespace speaker_id

