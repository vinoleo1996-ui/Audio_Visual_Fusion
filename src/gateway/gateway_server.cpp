#include "speaker_id/core/gateway_server.hpp"
#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <iomanip>

namespace speaker_id {

namespace {

// Base64 decoder implementation
[[maybe_unused]] std::vector<uint8_t> base64_decode(const std::string& in) {
  std::vector<uint8_t> out;
  std::vector<int> T(256, -1);
  const std::string b64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (int i = 0; i < 64; i++) {
    T[static_cast<unsigned char>(b64_chars[i])] = i;
  }
  int val = 0, valb = -8;
  for (unsigned char c : in) {
    if (T[c] == -1) continue;
    val = (val << 6) + T[c];
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

std::string EscapeJsonString(const std::string& input) {
  std::stringstream ss;
  for (char ch : input) {
    switch (ch) {
      case '\\': ss << "\\\\"; break;
      case '"':  ss << "\\\""; break;
      case '\b': ss << "\\b"; break;
      case '\f': ss << "\\f"; break;
      case '\n': ss << "\\n"; break;
      case '\r': ss << "\\r"; break;
      case '\t': ss << "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 32) {
          ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(ch));
        } else {
          ss << ch;
        }
        break;
    }
  }
  return ss.str();
}

} // namespace

std::string GatewayServer::SerializeFusionEvent(const FusionEvent& ev) const {
  std::lock_guard<std::mutex> lock(gallery_mu_);
  
  std::string type = ev.text.empty() ? "asd_update" : (ev.final ? "fusion" : "asr_partial");
  
  // Determine speaker_id and speaker_name
  std::string speaker_id = ev.speaker_id;
  std::string speaker_name = ev.speaker_name;
  
  if (speaker_id.empty()) {
    speaker_id = "offscreen";
    speaker_name = "画外";
    
    if (ev.position == "overlap") {
      speaker_id = "overlap";
      speaker_name = "多人";
    } else if (!ev.person_track_ids.empty()) {
      int primary_pid = ev.person_track_ids.front();
      std::optional<int> primary_fid;
      for (const auto& t : ev.tracks) {
        if (t.person_track_id == primary_pid) {
          primary_fid = t.face_track_id;
          break;
        }
      }
      
      if (primary_fid) {
        int fid = *primary_fid;
        speaker_id = "I" + std::to_string(fid);
        if (face_gallery_.count(fid) > 0) {
          speaker_name = face_gallery_.at(fid).name;
        } else {
          speaker_name = "I" + std::to_string(fid);
        }
      } else {
        speaker_id = "P" + std::to_string(primary_pid);
        speaker_name = "P" + std::to_string(primary_pid);
      }
    }
  } else {
    // Resolve friendly name for registered faces if diarization mapped it to a face track
    if (speaker_id.rfind("I", 0) == 0 && speaker_id.size() > 1) {
      try {
        int fid = std::stoi(speaker_id.substr(1));
        if (face_gallery_.count(fid) > 0) {
          speaker_name = face_gallery_.at(fid).name;
        }
      } catch (...) {}
    }
  }

  std::stringstream ss;
  ss << "{"
     << "\"type\":\"" << type << "\","
     << "\"utterance_id\":\"" << ev.utterance_id << "\","
     << "\"speaker_id\":\"" << speaker_id << "\","
     << "\"speaker_name\":\"" << EscapeJsonString(speaker_name) << "\","
     << "\"start_ms\":" << ev.start_ms << ","
     << "\"end_ms\":" << ev.end_ms << ","
     << "\"text\":\"" << EscapeJsonString(ev.text) << "\","
     << "\"text_delta\":\"" << EscapeJsonString(ev.text_delta) << "\","
     << "\"text_revision\":\"" << EscapeJsonString(ev.text_revision) << "\","
     << "\"final\":" << (ev.final ? "true" : "false") << ","
     << "\"position\":\"" << ev.position << "\","
     << "\"person_track_ids\":[";
  for (size_t i = 0; i < ev.person_track_ids.size(); ++i) {
    if (i > 0) ss << ",";
    ss << ev.person_track_ids[i];
  }
  ss << "],";
  ss << "\"confidence\":" << ev.confidence << ",";
  ss << "\"stability\":" << ev.stability << ",";
  ss << "\"token_timestamps_s\":[";
  for (size_t i = 0; i < ev.token_timestamps_s.size(); ++i) {
    if (i > 0) ss << ",";
    ss << ev.token_timestamps_s[i];
  }
  ss << "],";
  ss << "\"tentative\":" << (ev.tentative ? "true" : "false") << ",";
  ss << "\"tracks\":[";
  for (size_t i = 0; i < ev.tracks.size(); ++i) {
    if (i > 0) ss << ",";
    const auto& t = ev.tracks[i];
    
    BBox face_box = t.render.bbox;
    
    ss << "{"
       << "\"person_track_id\":" << t.person_track_id << ",";
    if (t.face_track_id) {
      int fid = *t.face_track_id;
      ss << "\"face_track_id\":" << fid << ",";
      if (face_gallery_.count(fid) > 0) {
        ss << "\"name\":\"" << EscapeJsonString(face_gallery_.at(fid).name) << "\",";
        ss << "\"background\":\"" << EscapeJsonString(face_gallery_.at(fid).background) << "\",";
      } else {
        ss << "\"name\":\"I" << fid << "\",";
        ss << "\"background\":\"\",";
      }
    } else {
      ss << "\"face_track_id\":null,";
      ss << "\"name\":\"P" << t.person_track_id << "\",";
      ss << "\"background\":\"\",";
    }
    ss << "\"bbox\":{\"x1\":" << face_box.x1 << ",\"y1\":" << face_box.y1 << ",\"x2\":" << face_box.x2 << ",\"y2\":" << face_box.y2 << "},"
       << "\"quality\":\"" << (t.quality == Quality::kGood ? "good" : (t.quality == Quality::kOk ? "ok" : "low")) << "\","
       << "\"p_active\":" << t.p_active << ","
       << "\"identity_id\":\"" << t.identity_id << "\","
       << "\"identity_name\":\"" << EscapeJsonString(t.identity_name) << "\","
       << "\"identity_state\":\"" << t.identity_state << "\","
       << "\"identity_confidence\":" << t.identity_confidence << ",";

    ss << "\"body\":{\"bbox\":{\"x1\":" << t.body.bbox.x1 << ",\"y1\":" << t.body.bbox.y1 << ",\"x2\":" << t.body.bbox.x2 << ",\"y2\":" << t.body.bbox.y2 << "}},";
    
    ss << "\"face\":{"
       << "\"face_bbox_observed\":" << (t.face.face_bbox_observed ? "true" : "false") << ","
       << "\"quality_score\":" << t.face.quality_score << ","
       << "\"bbox\":{\"x1\":" << t.face.bbox.x1 << ",\"y1\":" << t.face.bbox.y1 << ",\"x2\":" << t.face.bbox.x2 << ",\"y2\":" << t.face.bbox.y2 << "},"
       << "\"landmarks_5pt\":[";
    for (size_t k = 0; k < t.face.landmarks_5pt.size(); ++k) {
      if (k > 0) ss << ",";
      ss << "[" << t.face.landmarks_5pt[k].first << "," << t.face.landmarks_5pt[k].second << "]";
    }
    ss << "]"
       << "},";
       
    ss << "\"render\":{"
       << "\"bbox\":{\"x1\":" << t.render.bbox.x1 << ",\"y1\":" << t.render.bbox.y1 << ",\"x2\":" << t.render.bbox.x2 << ",\"y2\":" << t.render.bbox.y2 << "},"
       << "\"label\":\"" << EscapeJsonString(t.render.label) << "\","
       << "\"color_state\":\"" << t.render.color_state << "\","
       << "\"show_glow\":" << (t.render.show_glow ? "true" : "false")
       << "}";

    ss << "}";
  }
  ss << "]";
  ss << "}";
  return ss.str();
}


GatewayServer::GatewayServer(AppConfig config, std::shared_ptr<StreamingPipeline> pipeline)
    : config_(std::move(config)), pipeline_(std::move(pipeline)) {
  face_gallery_path_ = "models/face_gallery.json";
  
  int ws_port = config_.services.count("video") > 0 ? config_.services.at("video").port : 7200;
  ws_server_ = std::make_unique<WebSocketServer>("0.0.0.0", ws_port);
}

GatewayServer::~GatewayServer() {
  Stop();
}

bool GatewayServer::Start() {
  if (running_.exchange(true)) {
    return false;
  }

  LoadFaceGallery();
  ws_server_->Start();

  http_server_ = std::make_unique<httplib::Server>();
  http_thread_ = std::thread(&GatewayServer::HttpThreadLoop, this);
  std::cout << "C++ GatewayServer started on API port: " << Port() << "\n";
  return true;
}

void GatewayServer::Stop() {
  if (!running_.exchange(false)) {
    return;
  }

  ws_server_->Stop();
  if (http_server_) {
    http_server_->stop();
  }

  if (http_thread_.joinable()) {
    http_thread_.join();
  }
}

void GatewayServer::BroadcastVideoFrame(const std::vector<uint8_t>& jpeg_bytes) {
  if (running_) {
    ws_server_->BroadcastBinary(jpeg_bytes);
  }
}

void GatewayServer::ReportCameraStatus(
    bool ok, const std::string& error, const std::string& runtime_backend) {
  std::lock_guard<std::mutex> lock(status_mu_);
  camera_ok_ = ok;
  if (!runtime_backend.empty()) {
    camera_runtime_backend_ = runtime_backend;
  }
  if (!error.empty()) {
    latest_error_ = error;
  }
}

void GatewayServer::ReportAudioStatus(
    bool ok, const std::string& error, const std::string& runtime_backend) {
  std::lock_guard<std::mutex> lock(status_mu_);
  audio_ok_ = ok;
  if (!runtime_backend.empty()) {
    audio_runtime_backend_ = runtime_backend;
  }
  if (!error.empty()) {
    latest_error_ = error;
  }
}

void GatewayServer::ReportFatalError(const std::string& error) {
  std::lock_guard<std::mutex> lock(status_mu_);
  latest_error_ = error;
  fatal_error_ = error;
}

void GatewayServer::LoadFaceGallery() {
  std::lock_guard<std::mutex> lock(gallery_mu_);
  std::ifstream in(face_gallery_path_);
  if (!in) {
    std::cout << "No existing gallery found at " << face_gallery_path_ << ", starting fresh.\n";
    return;
  }
  
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  size_t pos = 0;
  while (true) {
    size_t key_pos = content.find('"', pos);
    if (key_pos == std::string::npos) break;
    size_t key_end = content.find('"', key_pos + 1);
    if (key_end == std::string::npos) break;
    
    std::string key = content.substr(key_pos + 1, key_end - key_pos - 1);
    pos = key_end + 1;
    
    bool is_numeric = !key.empty() && std::all_of(key.begin(), key.end(), ::isdigit);
    if (is_numeric) {
      int id = std::stoi(key);
      
      size_t obj_start = content.find('{', pos);
      size_t obj_end = content.find('}', obj_start + 1);
      if (obj_start == std::string::npos || obj_end == std::string::npos) break;
      
      std::string obj_str = content.substr(obj_start, obj_end - obj_start + 1);
      pos = obj_end + 1;
      
      GalleryItem item;
      
      size_t name_pos = obj_str.find("\"name\"");
      if (name_pos != std::string::npos) {
        size_t val_start = obj_str.find('"', name_pos + 6);
        size_t val_end = obj_str.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          item.name = obj_str.substr(val_start + 1, val_end - val_start - 1);
        }
      }
      
      size_t bg_pos = obj_str.find("\"background\"");
      if (bg_pos != std::string::npos) {
        size_t val_start = obj_str.find('"', bg_pos + 12);
        size_t val_end = obj_str.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          item.background = obj_str.substr(val_start + 1, val_end - val_start - 1);
        }
      }
      
      size_t emb_pos = obj_str.find("\"embedding\"");
      if (emb_pos != std::string::npos) {
        size_t arr_start = obj_str.find('[', emb_pos);
        size_t arr_end = obj_str.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
          std::string arr_str = obj_str.substr(arr_start + 1, arr_end - arr_start - 1);
          std::stringstream ss(arr_str);
          std::string val;
          while (std::getline(ss, val, ',')) {
            if (!val.empty()) {
              item.embedding.push_back(std::stof(val));
            }
          }
        }
      }
      
      face_gallery_[id] = item;
    }
  }
  std::cout << "Loaded " << face_gallery_.size() << " custom face identities from " << face_gallery_path_ << "\n";
}

void GatewayServer::SaveFaceGallery() {
  std::lock_guard<std::mutex> lock(gallery_mu_);
  SaveFaceGalleryLocked();
}

void GatewayServer::SaveFaceGalleryLocked() {
  std::ofstream out(face_gallery_path_);
  if (!out) {
    std::cerr << "Failed to save face gallery to " << face_gallery_path_ << "\n";
    return;
  }
  out << "{\n";
  bool first = true;
  for (const auto& item : face_gallery_) {
    if (!first) out << ",\n";
    first = false;
    out << "  \"" << item.first << "\": {\n"
        << "    \"name\": \"" << item.second.name << "\",\n"
        << "    \"background\": \"" << item.second.background << "\",\n"
        << "    \"embedding\": [";
    for (size_t i = 0; i < item.second.embedding.size(); ++i) {
      if (i > 0) out << ",";
      out << item.second.embedding[i];
    }
    out << "]\n"
        << "  }";
  }
  out << "\n}\n";
  out.close();
  if (pipeline_ && pipeline_->GetFaceEngine()) {
    pipeline_->GetFaceEngine()->ReloadGallery(face_gallery_path_);
  }
}

void GatewayServer::HttpThreadLoop() {
  using namespace httplib;
  auto& svr = *http_server_;

  // Serve static UI assets
  svr.set_mount_point("/ui", "./ui");
  svr.Get("/", [](const Request&, Response& res) {
    res.set_redirect("/ui/");
  });

  // REST endpoints
  svr.Get("/health", [this](const Request&, Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    std::lock_guard<std::mutex> lock(status_mu_);
    const auto pipeline_error = pipeline_->GetLatestError();
    const auto latest_error = latest_error_.empty() ? pipeline_error : latest_error_;
    std::stringstream ss;
    ss << "{"
       << "\"ok\":" << (latest_error.empty() && fatal_error_.empty() ? "true" : "false") << ","
       << "\"camera_ok\":" << (camera_ok_ ? "true" : "false") << ","
       << "\"audio_ok\":" << (audio_ok_ ? "true" : "false") << ","
       << "\"latest_error\":\"" << EscapeJsonString(latest_error) << "\","
       << "\"asr_backend\":\"" << config_.asr.backend << "\","
       << "\"asd_backend\":\"" << config_.asd.backend << "\","
       << "\"vad_backend\":\"" << config_.vad.backend << "\""
       << "}";
    res.set_content(ss.str(), "application/json");
  });

  svr.Get("/v2/status", [this](const Request&, Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    const auto tracks = pipeline_->GetCurrentTracks();
    const auto pipeline_error = pipeline_->GetLatestError();
    std::lock_guard<std::mutex> lock(status_mu_);
    const auto latest_error = latest_error_.empty() ? pipeline_error : latest_error_;
    std::stringstream ss;
    ss << "{"
       << "\"schema_version\":\"2.0\","
       << "\"health\":{"
       << "\"ok\":" << (latest_error.empty() && fatal_error_.empty() ? "true" : "false") << ","
       << "\"camera_ok\":" << (camera_ok_ ? "true" : "false") << ","
       << "\"audio_ok\":" << (audio_ok_ ? "true" : "false") << ","
       << "\"latest_error\":\"" << EscapeJsonString(latest_error) << "\","
       << "\"fatal_error\":\"" << EscapeJsonString(fatal_error_) << "\""
       << "},"
       << "\"providers\":{"
       << "\"video_capture\":\""
       << EscapeJsonString(camera_runtime_backend_.empty() ? "initializing"
                                                           : camera_runtime_backend_)
       << "\","
       << "\"audio_capture\":\""
       << EscapeJsonString(audio_runtime_backend_.empty() ? "initializing"
                                                          : audio_runtime_backend_)
       << "\","
       << "\"person_detector\":\"" << EscapeJsonString(config_.video.person_detector_backend) << "\","
       << "\"execution_provider\":\"cpu\","
       << "\"face\":\"" << EscapeJsonString(config_.face.backend) << "\","
       << "\"vad\":\"" << EscapeJsonString(config_.vad.backend) << "\","
       << "\"asr_final\":\"" << EscapeJsonString(config_.asr.backend) << "\","
       << "\"asr_partial\":\"" << EscapeJsonString(config_.asr.streaming_backend) << "\","
       << "\"asd\":\"" << EscapeJsonString(config_.asd.backend) << "\""
       << "},"
       << "\"fusion\":{"
       << "\"visible_tracks\":" << tracks.size() << ","
       << "\"multi_label_asd\":true,"
       << "\"tentative_results\":true,"
       << "\"llm_commit_requires_final\":true"
       << "},"
       << "\"sync\":{"
       << "\"timestamp_domain\":\"monotonic_ms\","
       << "\"video_time_offset_ms\":" << config_.sync.video_time_offset_ms
       << "},"
       << "\"latency_budget_ms\":{"
       << "\"binder_provisional_p95\":500,"
       << "\"turn_final_p95\":1500"
       << "}"
       << "}";
    res.set_content(ss.str(), "application/json");
  });

  svr.Get("/v1/status", [this](const Request&, Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    auto tracks = pipeline_->GetCurrentTracks();
    bool camera_ok = false;
    bool audio_ok = false;
    std::string latest_error;
    std::string fatal_error;
    const auto pipeline_error = pipeline_->GetLatestError();
    {
      std::lock_guard<std::mutex> lock(status_mu_);
      camera_ok = camera_ok_;
      audio_ok = audio_ok_;
      latest_error = latest_error_.empty() ? pipeline_error : latest_error_;
      fatal_error = fatal_error_;
    }
    
    std::stringstream ss;
    ss << "{"
       << "\"camera_ok\":" << (camera_ok ? "true" : "false") << ","
       << "\"audio_ok\":" << (audio_ok ? "true" : "false") << ","
       << "\"camera_index\":" << config_.video.camera_index << ","
       << "\"video_backend\":\"" << config_.video.camera_backend << "\","
       << "\"face_backend\":\"" << config_.face.backend << "\","
       << "\"audio_backend\":\"" << config_.audio.source << "\","
       << "\"vad_backend\":\"" << config_.vad.backend << "\","
       << "\"asr_backend\":\"" << config_.asr.backend << "\","
       << "\"asd_backend\":\"" << config_.asd.backend << "\","
       << "\"width\":" << config_.video.width << ","
       << "\"height\":" << config_.video.height << ","
       << "\"render_bbox\":" << (config_.video.render_bbox ? "true" : "false") << ","
       << "\"tracks\":[";
    
    {
      std::lock_guard<std::mutex> lock(gallery_mu_);
      for (size_t i = 0; i < tracks.size(); ++i) {
        if (i > 0) ss << ",";
        const auto& t = tracks[i];
        
        BBox face_box = t.render.bbox;
        
        ss << "{"
           << "\"person_track_id\":" << t.person_track_id << ",";
        if (t.face_track_id) {
          int fid = *t.face_track_id;
          ss << "\"face_track_id\":" << fid << ",";
          if (face_gallery_.count(fid) > 0) {
            ss << "\"name\":\"" << EscapeJsonString(face_gallery_.at(fid).name) << "\",";
            ss << "\"background\":\"" << EscapeJsonString(face_gallery_.at(fid).background) << "\"";
          } else {
            ss << "\"name\":\"I" << fid << "\",";
            ss << "\"background\":\"\"";
          }
        } else {
          ss << "\"face_track_id\":null,";
          ss << "\"name\":\"P" << t.person_track_id << "\",";
          ss << "\"background\":\"\"";
        }
        ss << ",\"bbox\":{\"x1\":" << face_box.x1 << ",\"y1\":" << face_box.y1 << ",\"x2\":" << face_box.x2 << ",\"y2\":" << face_box.y2 << "},";
        ss << "\"quality\":\"" << (t.quality == Quality::kGood ? "good" : (t.quality == Quality::kOk ? "ok" : "low")) << "\",";
        ss << "\"confidence\":" << t.confidence << ",";
        
        ss << "\"identity_id\":\"" << t.identity_id << "\",";
        ss << "\"identity_name\":\"" << EscapeJsonString(t.identity_name) << "\",";
        ss << "\"identity_state\":\"" << t.identity_state << "\",";
        ss << "\"identity_confidence\":" << t.identity_confidence << ",";

        ss << "\"body\":{\"bbox\":{\"x1\":" << t.body.bbox.x1 << ",\"y1\":" << t.body.bbox.y1 << ",\"x2\":" << t.body.bbox.x2 << ",\"y2\":" << t.body.bbox.y2 << "}},";
        
        ss << "\"face\":{"
           << "\"face_bbox_observed\":" << (t.face.face_bbox_observed ? "true" : "false") << ","
           << "\"quality_score\":" << t.face.quality_score << ","
           << "\"bbox\":{\"x1\":" << t.face.bbox.x1 << ",\"y1\":" << t.face.bbox.y1 << ",\"x2\":" << t.face.bbox.x2 << ",\"y2\":" << t.face.bbox.y2 << "}"
           << "},";
           
        ss << "\"render\":{"
           << "\"bbox\":{\"x1\":" << t.render.bbox.x1 << ",\"y1\":" << t.render.bbox.y1 << ",\"x2\":" << t.render.bbox.x2 << ",\"y2\":" << t.render.bbox.y2 << "},"
           << "\"label\":\"" << EscapeJsonString(t.render.label) << "\","
           << "\"color_state\":\"" << t.render.color_state << "\","
           << "\"show_glow\":" << (t.render.show_glow ? "true" : "false")
           << "}";
           
        ss << "}";
      }
    }
    
    ss << "],"
       << "\"speaker_state\":{\"current_speakers\":[],\"speaking\":false},"
       << "\"latest_error\":\"" << EscapeJsonString(latest_error) << "\","
       << "\"fatal_error\":\"" << EscapeJsonString(fatal_error) << "\""
       << "}";
    res.set_content(ss.str(), "application/json");
  });

  svr.Get("/v1/gallery", [this](const Request&, Response& res) {
    std::lock_guard<std::mutex> lock(gallery_mu_);
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const auto& item : face_gallery_) {
      if (!first) ss << ",";
      first = false;
      ss << "\"" << item.first << "\":{"
         << "\"name\":\"" << EscapeJsonString(item.second.name) << "\","
         << "\"background\":\"" << EscapeJsonString(item.second.background) << "\""
         << "}";
    }
    ss << "}";
    res.set_content(ss.str(), "application/json");
    res.set_header("Access-Control-Allow-Origin", "*");
  });

  svr.Post("/v1/gallery/upload", [this](const Request& req, Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    try {
      std::string body = req.body;
      std::string name;
      size_t name_pos = body.find("\"name\"");
      if (name_pos != std::string::npos) {
        size_t val_start = body.find('"', name_pos + 6);
        size_t val_end = body.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          name = body.substr(val_start + 1, val_end - val_start - 1);
        }
      }
      
      std::string background;
      size_t bg_pos = body.find("\"background\"");
      if (bg_pos != std::string::npos) {
        size_t val_start = body.find('"', bg_pos + 12);
        size_t val_end = body.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          background = body.substr(val_start + 1, val_end - val_start - 1);
        }
      }

      std::string image;
      size_t img_pos = body.find("\"image\"");
      if (img_pos != std::string::npos) {
        size_t val_start = body.find('"', img_pos + 7);
        size_t val_end = body.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          image = body.substr(val_start + 1, val_end - val_start - 1);
        }
      }

      if (name.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"Missing person name\"}", "application/json");
        return;
      }

      std::vector<float> embedding;
      std::vector<uint8_t> decoded_img;
      if (!image.empty()) {
        size_t comma = image.find(',');
        std::string b64_data = (comma != std::string::npos) ? image.substr(comma + 1) : image;
        decoded_img = base64_decode(b64_data);
      }

      if (!decoded_img.empty() && pipeline_ && pipeline_->GetFaceEngine()) {
        cv::Mat mat = cv::imdecode(decoded_img, cv::IMREAD_COLOR);
        if (!mat.empty()) {
          auto face_detections = pipeline_->GetFaceEngine()->DetectFaces(mat, 0.45F);
          if (!face_detections.empty()) {
            embedding = pipeline_->GetFaceEngine()->ExtractEmbedding(mat, face_detections[0].landmarks);
          }
        }
      }
      if (embedding.size() != 512) {
        res.status = 400;
        res.set_content("{\"error\":\"Uploaded image must contain one detectable face\"}", "application/json");
        return;
      }

      std::lock_guard<std::mutex> lock(gallery_mu_);
      int next_id = 1000;
      if (!face_gallery_.empty()) {
        next_id = face_gallery_.rbegin()->first + 1;
      }
      
      GalleryItem item;
      item.name = name;
      item.background = background;
      item.embedding = embedding;
      face_gallery_[next_id] = item;
      
      SaveFaceGalleryLocked();
      
      std::stringstream ss;
      ss << "{\"ok\":true,\"id\":" << next_id << ",\"name\":\"" << name << "\"}";
      res.set_content(ss.str(), "application/json");
    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
    }
  });

  svr.Post("/v1/gallery/update", [this](const Request& req, Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    try {
      std::string body = req.body;
      int id = -1;
      size_t id_pos = body.find("\"id\"");
      if (id_pos != std::string::npos) {
        size_t val_start = body.find_first_of("0123456789", id_pos + 4);
        if (val_start != std::string::npos) {
          id = std::stoi(body.substr(val_start));
        }
      }
      
      std::string name;
      size_t name_pos = body.find("\"name\"");
      if (name_pos != std::string::npos) {
        size_t val_start = body.find('"', name_pos + 6);
        size_t val_end = body.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          name = body.substr(val_start + 1, val_end - val_start - 1);
        }
      }
      
      std::string background;
      size_t bg_pos = body.find("\"background\"");
      if (bg_pos != std::string::npos) {
        size_t val_start = body.find('"', bg_pos + 12);
        size_t val_end = body.find('"', val_start + 1);
        if (val_start != std::string::npos && val_end != std::string::npos) {
          background = body.substr(val_start + 1, val_end - val_start - 1);
        }
      }

      std::lock_guard<std::mutex> lock(gallery_mu_);
      if (face_gallery_.count(id) > 0) {
        if (!name.empty()) {
          face_gallery_[id].name = name;
        }
        face_gallery_[id].background = background;
        SaveFaceGalleryLocked();
        res.set_content("{\"ok\":true}", "application/json");
      } else {
        res.status = 404;
        res.set_content("{\"error\":\"Identity not found\"}", "application/json");
      }
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
    }
  });

  svr.Post("/v1/gallery/delete", [this](const Request& req, Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    try {
      std::string body = req.body;
      int id = -1;
      size_t id_pos = body.find("\"id\"");
      if (id_pos != std::string::npos) {
        size_t val_start = body.find_first_of("0123456789", id_pos + 4);
        if (val_start != std::string::npos) {
          id = std::stoi(body.substr(val_start));
        }
      }

      std::lock_guard<std::mutex> lock(gallery_mu_);
      if (face_gallery_.count(id) > 0) {
        face_gallery_.erase(id);
        SaveFaceGalleryLocked();
        res.set_content("{\"ok\":true}", "application/json");
      } else {
        res.status = 404;
        res.set_content("{\"error\":\"Identity not found\"}", "application/json");
      }
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
    }
  });

  // Server-Sent Events Endpoint
  auto fusion_events_handler = [this](const Request&, Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    
    res.set_chunked_content_provider(
        "text/event-stream; charset=utf-8",
        [this](size_t offset, DataSink &sink) {
          if (!sink.is_writable()) {
            return false;
          }
          if (offset == 0) {
            bool camera_ok = false;
            bool audio_ok = false;
            {
              std::lock_guard<std::mutex> lock(status_mu_);
              camera_ok = camera_ok_;
              audio_ok = audio_ok_;
            }
            std::stringstream hello;
            hello << "data: {"
                  << "\"type\":\"hello\","
                  << "\"camera_ok\":" << (camera_ok ? "true" : "false") << ","
                  << "\"audio_ok\":" << (audio_ok ? "true" : "false") << ","
                  << "\"width\":" << config_.video.width << ","
                  << "\"height\":" << config_.video.height << ","
                  << "\"asd_active_threshold\":" << config_.asd.speaking_threshold << ","
                  << "\"render_bbox\":" << (config_.video.render_bbox ? "true" : "false") << ","
                  << "\"vad_backend\":\"" << config_.vad.backend << "\""
                  << "}\n\n";
            std::string hello_str = hello.str();
            if (!sink.write(hello_str.data(), hello_str.size())) {
              return false;
            }
          }
          
          auto events = pipeline_->GetLatestEvents();
          if (events.empty()) {
            std::stringstream hb;
            hb << "data: {\"type\":\"heartbeat\",\"time_ms\":" 
               << std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count()
               << "}\n\n";
            std::string hb_str = hb.str();
            if (!sink.write(hb_str.data(), hb_str.size())) {
              return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
          } else {
            for (const auto& ev : events) {
              std::string json_str = SerializeFusionEvent(ev);
              std::string sse_evt = "data: " + json_str + "\n\n";
              if (!sink.write(sse_evt.data(), sse_evt.size())) {
                break;
              }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          return running_.load();
        }
    );
  };
  svr.Get("/v1/fusion/events", fusion_events_handler);
  svr.Get("/v2/fusion/events", fusion_events_handler);

  svr.listen("0.0.0.0", Port());
}

} // namespace speaker_id
