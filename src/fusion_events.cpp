#include "speaker_id/modules/fusion.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace speaker_id {
namespace {

const UtteranceEvent* FindUtterance(
    const std::vector<UtteranceEvent>& utterances,
    const std::string& utterance_id) {
  for (const auto& utterance : utterances) {
    if (utterance.utterance_id == utterance_id) {
      return &utterance;
    }
  }
  return nullptr;
}

float ActiveScoreForPerson(const std::vector<ActiveSpeakerScore>& scores, int person_track_id) {
  float best = 0.0F;
  for (const auto& score : scores) {
    if (score.person_track_id == person_track_id) {
      best = std::max(best, score.p_active);
    }
  }
  return best;
}

}  // namespace

RuleFusionBackend::RuleFusionBackend(
    int frame_width,
    float active_threshold,
    float visible_threshold)
    : frame_width_(frame_width),
      active_threshold_(active_threshold),
      visible_threshold_(visible_threshold) {}

std::vector<SpeakerAttribution> RuleFusionBackend::Fuse(const FusionInput& input) {
  std::vector<SpeakerAttribution> results;
  for (const auto& utterance : input.utterances) {
    if (!HasVadEvidence(input, utterance)) {
      continue;
    }

    SpeakerAttribution attribution;
    attribution.utterance_id = utterance.utterance_id;
    attribution.tentative = !utterance.final;

    std::vector<std::pair<float, const TrackEvent*>> scored;
    scored.reserve(input.tracks.size());
    for (const auto& track : input.tracks) {
      scored.push_back({ScoreForPerson(input, track.person_track_id), &track});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& left, const auto& right) {
      return left.first > right.first;
    });

    std::vector<const TrackEvent*> active;
    for (const auto& item : scored) {
      if (item.first >= active_threshold_) {
        active.push_back(item.second);
      }
    }

    if (active.size() >= 2U) {
      attribution.position = "overlap";
      attribution.person_track_ids = {active[0]->person_track_id, active[1]->person_track_id};
      attribution.confidence = scored.empty() ? 0.0F : scored.front().first;
    } else if (active.size() == 1U) {
      attribution.position = PositionForTrack(*active[0]);
      attribution.person_track_ids = {active[0]->person_track_id};
      attribution.confidence = scored.empty() ? 0.0F : scored.front().first;
    } else if (!scored.empty() && scored.front().first >= visible_threshold_) {
      attribution.position = PositionForTrack(*scored.front().second);
      attribution.person_track_ids = {scored.front().second->person_track_id};
      attribution.confidence = scored.front().first;
    } else {
      attribution.position = "offscreen";
      attribution.confidence = 0.72F;
    }

    results.push_back(attribution);
  }
  return results;
}

std::string RuleFusionBackend::PositionForTrack(const TrackEvent& track) const {
  const float center_x = (track.bbox.x1 + track.bbox.x2) * 0.5F;
  if (center_x < static_cast<float>(frame_width_) * 0.42F) {
    return "left";
  }
  if (center_x > static_cast<float>(frame_width_) * 0.58F) {
    return "right";
  }
  return "center";
}

float RuleFusionBackend::ScoreForPerson(const FusionInput& input, int person_track_id) const {
  float best = 0.0F;
  for (const auto& score : input.active_speaker_scores) {
    if (score.person_track_id == person_track_id) {
      best = std::max(best, score.p_active);
    }
  }
  return best;
}

bool RuleFusionBackend::HasVadEvidence(const FusionInput& input, const UtteranceEvent& utterance) const {
  if (!utterance.final) {
    return true;
  }
  for (const auto& vad : input.vad_segments) {
    if (vad.end_ms >= utterance.start_ms && vad.start_ms <= utterance.end_ms) {
      return true;
    }
  }
  return false;
}

std::vector<FusionEvent> BuildFusionEvents(
    const FusionInput& input,
    const std::vector<SpeakerAttribution>& attributions) {
  std::vector<FusionEvent> events;
  for (const auto& attribution : attributions) {
    const auto* utterance = FindUtterance(input.utterances, attribution.utterance_id);
    if (utterance == nullptr) {
      continue;
    }

    FusionEvent event;
    event.utterance_id = utterance->utterance_id;
    event.start_ms = utterance->start_ms;
    event.end_ms = utterance->end_ms;
    event.text = utterance->text;
    event.text_delta = utterance->text_delta;
    event.text_revision = utterance->text_revision;
    event.final = utterance->final;
    event.position = attribution.position;
    event.person_track_ids = attribution.person_track_ids;
    event.confidence = attribution.confidence;
    event.stability = utterance->stability;
    event.token_timestamps_s = utterance->token_timestamps_s;
    event.tentative = attribution.tentative;

    for (const auto& track : input.tracks) {
      FusionTrackView view;
      view.person_track_id = track.person_track_id;
      view.face_track_id = track.face_track_id;
      view.bbox = track.bbox;
      view.quality = track.quality;
      view.p_active = ActiveScoreForPerson(input.active_speaker_scores, track.person_track_id);
      
      view.face = track.face;
      view.body = track.body;
      view.render = track.render;
      view.identity_id = track.identity_id;
      view.identity_name = track.identity_name;
      view.identity_state = track.identity_state;
      view.identity_confidence = track.identity_confidence;

      event.tracks.push_back(view);
    }

    events.push_back(event);
  }
  return events;
}

}  // namespace speaker_id
