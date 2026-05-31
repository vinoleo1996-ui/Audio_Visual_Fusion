#pragma once

#include "speaker_id/core/types.hpp"

#include <string>

namespace speaker_id {

enum class LlmTurnAction {
  kIgnore,
  kPrefetch,
  kCommit,
};

struct LlmTurnDecision {
  LlmTurnAction action = LlmTurnAction::kIgnore;
  std::string utterance_id;
  std::string text;
  std::string speaker_id;
};

class LlmTurnGate {
 public:
  LlmTurnDecision Evaluate(const FusionEvent& event) const {
    LlmTurnDecision decision;
    decision.utterance_id = event.utterance_id;
    decision.text = event.text;
    decision.speaker_id = event.speaker_id;
    if (event.text.empty() || event.utterance_id.empty()) {
      return decision;
    }
    if (event.final && !event.tentative) {
      decision.action = LlmTurnAction::kCommit;
    } else {
      decision.action = LlmTurnAction::kPrefetch;
    }
    return decision;
  }
};

}  // namespace speaker_id
