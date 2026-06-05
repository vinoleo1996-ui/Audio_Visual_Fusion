#!/usr/bin/env python3
"""Local memory ingestion service for robot.memory.input.v1 envelopes."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.server
import json
import mimetypes
import os
import pathlib
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from typing import Any


INPUT_SCHEMA = "robot.memory.input.v1"
DEFAULT_DB_PATH = pathlib.Path("output/memory/memory.sqlite3")
DEFAULT_LLM_ENDPOINT = os.environ.get(
    "MEMORY_LLM_ENDPOINT", "http://127.0.0.1:8081/v1/chat/completions"
)
DEFAULT_LLM_MODEL = os.environ.get("MEMORY_LLM_MODEL", "qwen3.5-2b-q4_k_m")
DEFAULT_LLM_TIMEOUT_S = float(os.environ.get("MEMORY_LLM_TIMEOUT_S", "60"))
DEFAULT_MODEL_PATH = pathlib.Path(
    "/home/nvidia/Documents/Models/Qwen3.5-2B-Q4_K_M/Qwen3.5-2B-Q4_K_M.gguf"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "ui"

ATOM_TYPES = {
    "fact",
    "preference",
    "temporary_state",
    "plan",
    "commitment",
    "alias",
    "rule",
    "action_experience",
}
SUBJECT_SCOPES = {"person", "household", "group", "unknown", "robot"}
ATOM_STATUSES = {"active", "uncertain", "superseded", "deleted"}
SENSITIVITY_VALUES = {"normal", "sensitive"}
SPEAKER_STATES = {
    "known_visible_speaker",
    "known_offscreen_speaker",
    "uncertain",
    "overlap_speech",
}

CHINESE_PUNCT_RE = re.compile(r"[\s,.;:!?，。；：！？、~·`'\"“”‘’（）()\[\]{}<>《》|/\\_-]+")
NOISE_TEXTS = {
    "",
    "嗯",
    "嗯嗯",
    "啊",
    "哦",
    "额",
    "呃",
    "喂",
    "喂喂",
    "对",
    "对对",
    "好",
    "好的",
    "哈哈",
}
SENSITIVE_HINTS = (
    "身份证",
    "密码",
    "银行卡",
    "住址",
    "家庭住址",
    "手机号",
    "电话",
    "病历",
    "诊断",
    "药",
    "过敏",
    "收入",
    "工资",
    "宗教",
    "政治",
)
EXPLICIT_MEMORY_HINTS = ("记住", "记一下", "帮我记", "帮我记住", "以后记得", "remember")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    endpoint: str = DEFAULT_LLM_ENDPOINT
    model: str = DEFAULT_LLM_MODEL
    no_llm: bool = False
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    fallback_on_error: bool = False


def now_ms() -> int:
    return int(time.time() * 1000)


def json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=None, indent=indent)


def stable_id(prefix: str, *parts: Any, length: int = 18) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json_dumps(part).encode("utf-8"))
        digest.update(b"\x00")
    return f"{prefix}_{digest.hexdigest()[:length]}"


def get_path(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for key in path.split("."):
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if isinstance(current, list) and key.isdigit():
            index = int(key)
            if index >= len(current):
                return default
            current = current[index]
            continue
        return default
    return current


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def clamp01(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_json_or_stdin(path_text: str) -> dict[str, Any]:
    if path_text == "-":
        value = json.load(sys.stdin)
    else:
        value = load_json(pathlib.Path(path_text))
    if not isinstance(value, dict):
        raise ValueError("input must contain a JSON object")
    return value


def connect_db(path: pathlib.Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_memory_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            household_id TEXT NOT NULL,
            robot_id TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            subject_id TEXT NOT NULL,
            speaker_state TEXT NOT NULL,
            allow_memory_write INTEGER NOT NULL,
            allow_profile_write INTEGER NOT NULL,
            need_clarification INTEGER NOT NULL,
            profile_write_reason TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            slow_reaction_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_raw_memory_events_session_turn
            ON raw_memory_events(session_id, turn_id);
        CREATE INDEX IF NOT EXISTS idx_raw_memory_events_subject_ts
            ON raw_memory_events(subject_id, ts_ms);

        CREATE TABLE IF NOT EXISTS short_term_state (
            state_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            state_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_short_term_state_subject_type
            ON short_term_state(subject_id, state_type, created_at_ms);

        CREATE TABLE IF NOT EXISTS memory_atoms (
            memory_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            subject_scope TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            type TEXT NOT NULL,
            text TEXT NOT NULL,
            confidence REAL NOT NULL,
            importance REAL NOT NULL,
            valid_from_ms INTEGER NOT NULL,
            valid_until_ms INTEGER,
            sensitivity TEXT NOT NULL,
            evidence_event_ids TEXT NOT NULL,
            source_turn_id TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memory_atoms_subject
            ON memory_atoms(subject_scope, subject_id, type, status, updated_at_ms);
        CREATE INDEX IF NOT EXISTS idx_memory_atoms_event
            ON memory_atoms(event_id);

        CREATE TABLE IF NOT EXISTS person_profiles (
            person_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_edges (
            edge_id TEXT PRIMARY KEY,
            from_memory_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            to_memory_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS confirmation_candidates (
            candidate_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            prompt TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_call_logs (
            call_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            task TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            model TEXT NOT NULL,
            ok INTEGER NOT NULL,
            error TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_text TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        """
    )
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_atoms_fts
            USING fts5(memory_id UNINDEXED, text)
            """
        )
    except sqlite3.OperationalError:
        pass
    conn.commit()


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def compact_text(text: str) -> str:
    return CHINESE_PUNCT_RE.sub("", text).strip().lower()


def is_text_valid_for_memory(envelope: dict[str, Any]) -> bool:
    text = normalize_text(get_path(envelope, "utterance.text", ""))
    compact = compact_text(text)
    if compact in NOISE_TEXTS or len(compact) < 2:
        return False
    confidence = get_path(envelope, "utterance.asr_confidence")
    if isinstance(confidence, (int, float)) and confidence > 0 and confidence < 0.15:
        return False
    audio_quality = get_path(envelope, "fast_reaction.audio_event.audio_activity.asr_quality_reason")
    if isinstance(audio_quality, str) and audio_quality not in {"", "ok", "accepted"}:
        return False
    return True


def is_sensitive_text(text: str) -> bool:
    return any(hint in text for hint in SENSITIVE_HINTS)


def explicitly_requests_memory(text: str) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in EXPLICIT_MEMORY_HINTS)


def has_av_conflict(envelope: dict[str, Any]) -> bool:
    checks = [
        get_path(envelope, "fast_reaction.speaker_turn.evidence.conflict"),
        get_path(envelope, "fast_reaction.speaker_turn.evidence.mixed_speech_risk"),
        get_path(envelope, "fast_reaction.speaker_turn.audio.mixed_speech_risk"),
        get_path(envelope, "fast_reaction.audio_event.mixed_speech_risk"),
    ]
    return any(as_bool(value) for value in checks)


def has_bound_voiceprint(envelope: dict[str, Any]) -> bool:
    identity_context = get_path(envelope, "subject.identity_context", {}) or {}
    speaker_turn = get_path(envelope, "fast_reaction.speaker_turn", {}) or {}
    audio_event = get_path(envelope, "fast_reaction.audio_event", {}) or {}
    voiceprint_id = (
        identity_context.get("voiceprint_id")
        or speaker_turn.get("voiceprint_id")
        or get_path(speaker_turn, "audio.voiceprint_id")
        or audio_event.get("voiceprint_id")
    )
    states = {
        str(identity_context.get("spk_state", "")),
        str(identity_context.get("voiceprint_state", "")),
        str(speaker_turn.get("spk_state", "")),
        str(speaker_turn.get("voiceprint_state", "")),
        str(get_path(speaker_turn, "audio.spk_state", "")),
        str(get_path(speaker_turn, "audio.voiceprint_state", "")),
        str(audio_event.get("spk_state", "")),
        str(audio_event.get("voiceprint_state", "")),
    }
    return bool(voiceprint_id) and bool(
        states
        & {
            "spk_voiceprint_linked",
            "voiceprint_linked",
            "matched",
            "bound",
            "linked",
        }
    )


def envelope_policy(envelope: dict[str, Any]) -> dict[str, Any]:
    policy = get_path(envelope, "fast_reaction.policy", {}) or {}
    speaker_turn = get_path(envelope, "fast_reaction.speaker_turn", {}) or {}
    return {
        "allow_memory_write": as_bool(
            policy.get("allow_memory_write", speaker_turn.get("allow_memory_write"))
        ),
        "allow_profile_write": as_bool(
            policy.get("allow_profile_write", speaker_turn.get("allow_profile_write"))
        ),
        "need_clarification": as_bool(
            policy.get("need_clarification", get_path(speaker_turn, "policy.need_clarification"))
        ),
        "profile_write_reason": str(
            policy.get("profile_write_reason")
            or speaker_turn.get("profile_write_reason")
            or ""
        ),
    }


def evaluate_write_gate(envelope: dict[str, Any]) -> dict[str, Any]:
    policy = envelope_policy(envelope)
    speaker_state = str(get_path(envelope, "subject.speaker_state", "uncertain"))
    text = normalize_text(get_path(envelope, "utterance.text", ""))
    reasons: list[str] = []

    if not (policy["allow_memory_write"] or policy["allow_profile_write"]):
        reasons.append("policy_disallows_memory_write")

    if speaker_state == "overlap_speech":
        reasons.append("overlap_speech")
    elif speaker_state == "known_visible_speaker":
        pass
    elif speaker_state == "known_offscreen_speaker":
        if not has_bound_voiceprint(envelope):
            reasons.append("offscreen_without_bound_voiceprint")
    else:
        reasons.append("speaker_identity_not_committed")

    if has_av_conflict(envelope):
        reasons.append("av_conflict_or_mixed_speech")
    if policy["need_clarification"]:
        reasons.append("need_clarification")
    if not is_text_valid_for_memory(envelope):
        reasons.append("invalid_or_noisy_asr_text")
    if is_sensitive_text(text) and not explicitly_requests_memory(text):
        reasons.append("sensitive_without_explicit_memory_request")

    return {
        "allowed": not reasons,
        "reasons": reasons,
        "policy": policy,
        "speaker_state": speaker_state,
    }


def normalize_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    if envelope.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"expected schema {INPUT_SCHEMA}, got {envelope.get('schema')!r}")
    normalized = json.loads(json_dumps(envelope))
    normalized.setdefault("event_id", stable_id("mem_in", normalized))
    normalized.setdefault("session_id", "")
    normalized.setdefault("turn_id", "")
    normalized.setdefault("household_id", "local_household")
    normalized.setdefault("robot_id", "local_robot")
    normalized.setdefault("ts_ms", now_ms())
    normalized.setdefault("subject", {})
    normalized.setdefault("utterance", {})
    normalized.setdefault("fast_reaction", {})
    normalized.setdefault("slow_reaction", {})
    normalized.setdefault("robot_interaction", {})
    normalized.setdefault("source_refs", {})

    subject = normalized["subject"]
    subject.setdefault("person_id", "")
    subject.setdefault("display_name", "")
    subject.setdefault("speaker_state", "uncertain")
    subject.setdefault("identity_context", {})
    subject.setdefault("speaker_source", {})
    subject.setdefault("speaker_probabilities", [])
    if subject["speaker_state"] not in SPEAKER_STATES:
        subject["speaker_state"] = map_speaker_state({"speaker_state": subject["speaker_state"]})

    utterance = normalized["utterance"]
    utterance.setdefault("text", "")
    utterance.setdefault("language", "zh")
    utterance.setdefault("start_ms", 0)
    utterance.setdefault("end_ms", 0)
    utterance.setdefault("asr_confidence", 0.0)
    utterance.setdefault("segments", [])

    fast_reaction = normalized["fast_reaction"]
    fast_reaction.setdefault("speaker_turn", {})
    fast_reaction.setdefault("audio_event", {})
    fast_reaction.setdefault("visual", {})
    fast_reaction.setdefault("nlu", {})
    fast_reaction.setdefault("policy", {})
    existing_policy = (
        fast_reaction["policy"] if isinstance(fast_reaction["policy"], dict) else {}
    )
    policy = envelope_policy(normalized)
    fast_reaction["policy"] = {**existing_policy, **policy}

    slow_reaction = normalized["slow_reaction"]
    slow_reaction.setdefault("scene", {})
    slow_reaction.setdefault("environment", {})
    slow_reaction.setdefault("scene_model", "")
    scene = slow_reaction["scene"]
    scene.setdefault("summary", "")
    scene.setdefault("activity", "")
    scene.setdefault("objects", [])
    scene.setdefault("room", "")
    scene.setdefault("people", [])

    robot_interaction = normalized["robot_interaction"]
    robot_interaction.setdefault("reply_text", "")
    robot_interaction.setdefault("commands", [])
    robot_interaction.setdefault("scheduler_result", {})
    robot_interaction.setdefault("proactive_candidate", {})
    robot_interaction.setdefault("execution_result", {})

    source_refs = normalized["source_refs"]
    source_refs.setdefault("raw_event_ids", [])
    source_refs.setdefault("snapshot_refs", [])
    source_refs.setdefault("audio_refs", [])
    return normalized


def upsert_raw_event(conn: sqlite3.Connection, envelope: dict[str, Any]) -> bool:
    event_id = str(envelope["event_id"])
    existed = (
        conn.execute(
            "SELECT 1 FROM raw_memory_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        is not None
    )
    policy = envelope_policy(envelope)
    stamp = now_ms()
    conn.execute(
        """
        INSERT INTO raw_memory_events (
            event_id, session_id, turn_id, household_id, robot_id, ts_ms,
            subject_id, speaker_state, allow_memory_write, allow_profile_write,
            need_clarification, profile_write_reason, raw_json, slow_reaction_json,
            created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            session_id = excluded.session_id,
            turn_id = excluded.turn_id,
            household_id = excluded.household_id,
            robot_id = excluded.robot_id,
            ts_ms = excluded.ts_ms,
            subject_id = excluded.subject_id,
            speaker_state = excluded.speaker_state,
            allow_memory_write = excluded.allow_memory_write,
            allow_profile_write = excluded.allow_profile_write,
            need_clarification = excluded.need_clarification,
            profile_write_reason = excluded.profile_write_reason,
            raw_json = excluded.raw_json,
            slow_reaction_json = excluded.slow_reaction_json,
            updated_at_ms = excluded.updated_at_ms
        """,
        (
            event_id,
            str(envelope["session_id"]),
            str(envelope["turn_id"]),
            str(envelope["household_id"]),
            str(envelope["robot_id"]),
            int(envelope["ts_ms"]),
            str(get_path(envelope, "subject.person_id", "")),
            str(get_path(envelope, "subject.speaker_state", "uncertain")),
            int(policy["allow_memory_write"]),
            int(policy["allow_profile_write"]),
            int(policy["need_clarification"]),
            policy["profile_write_reason"],
            json_dumps(envelope),
            json_dumps(envelope.get("slow_reaction", {})),
            stamp,
            stamp,
        ),
    )
    return existed


def build_short_term_entries(
    envelope: dict[str, Any], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    event_id = str(envelope["event_id"])
    session_id = str(envelope["session_id"])
    turn_id = str(envelope["turn_id"])
    subject_id = str(get_path(envelope, "subject.person_id", "") or "unknown")
    text = normalize_text(get_path(envelope, "utterance.text", ""))
    display_name = normalize_text(get_path(envelope, "subject.display_name", "")) or subject_id

    if text:
        summary = f"{display_name}: {text}"
        entries.append(
            {
                "state_id": stable_id("st", event_id, "recent_turn"),
                "event_id": event_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "subject_id": subject_id,
                "state_type": "recent_turn",
                "summary": summary[:240],
                "payload": {
                    "utterance": envelope.get("utterance", {}),
                    "speaker_state": get_path(envelope, "subject.speaker_state", "uncertain"),
                    "scene": get_path(envelope, "slow_reaction.scene", {}),
                    "write_gate": gate,
                },
            }
        )

    commands = get_path(envelope, "robot_interaction.commands", []) or []
    if commands:
        summary = "robot commands: " + ", ".join(
            str(command.get("command", command.get("type", "unknown")))
            for command in commands
            if isinstance(command, dict)
        )
        entries.append(
            {
                "state_id": stable_id("st", event_id, "robot_commands"),
                "event_id": event_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "subject_id": subject_id,
                "state_type": "robot_commands",
                "summary": summary[:240],
                "payload": {"commands": commands},
            }
        )

    nlu = get_path(envelope, "fast_reaction.nlu", {}) or {}
    if isinstance(nlu, dict) and nlu:
        entries.append(
            {
                "state_id": stable_id("st", event_id, "nlu"),
                "event_id": event_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "subject_id": subject_id,
                "state_type": "task_slots",
                "summary": "current task slots",
                "payload": nlu,
            }
        )

    if not gate["allowed"] and (
        "speaker_identity_not_committed" in gate["reasons"]
        or "overlap_speech" in gate["reasons"]
        or "need_clarification" in gate["reasons"]
    ):
        prompt = build_confirmation_prompt(envelope, gate)
        entries.append(
            {
                "state_id": stable_id("st", event_id, "pending_confirmation"),
                "event_id": event_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "subject_id": subject_id,
                "state_type": "pending_confirmation",
                "summary": prompt,
                "payload": {"prompt": prompt, "reasons": gate["reasons"]},
            }
        )
    return entries


def insert_short_term_entries(
    conn: sqlite3.Connection, entries: list[dict[str, Any]]
) -> list[str]:
    stamp = now_ms()
    ids: list[str] = []
    for entry in entries:
        conn.execute(
            """
            INSERT INTO short_term_state (
                state_id, event_id, session_id, turn_id, subject_id,
                state_type, summary, payload_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(state_id) DO UPDATE SET
                summary = excluded.summary,
                payload_json = excluded.payload_json,
                updated_at_ms = excluded.updated_at_ms
            """,
            (
                entry["state_id"],
                entry["event_id"],
                entry["session_id"],
                entry["turn_id"],
                entry["subject_id"],
                entry["state_type"],
                entry["summary"],
                json_dumps(entry["payload"]),
                stamp,
                stamp,
            ),
        )
        ids.append(entry["state_id"])
    return ids


def build_confirmation_prompt(envelope: dict[str, Any], gate: dict[str, Any]) -> str:
    text = normalize_text(get_path(envelope, "utterance.text", ""))
    if "overlap_speech" in gate["reasons"]:
        return f"刚刚多人同时说话，哪一句需要记住：{text}"
    if "speaker_identity_not_committed" in gate["reasons"]:
        return f"刚刚是谁说的：{text}"
    if "need_clarification" in gate["reasons"]:
        return f"刚刚这句话是否需要记到某个人名下：{text}"
    return f"这条信息是否需要记住：{text}"


def insert_confirmation_candidate(
    conn: sqlite3.Connection, envelope: dict[str, Any], gate: dict[str, Any]
) -> str | None:
    if gate["allowed"]:
        return None
    interesting_reasons = {
        "speaker_identity_not_committed",
        "overlap_speech",
        "need_clarification",
        "offscreen_without_bound_voiceprint",
    }
    if not interesting_reasons.intersection(gate["reasons"]):
        return None
    prompt = build_confirmation_prompt(envelope, gate)
    event_id = str(envelope["event_id"])
    reason = ",".join(gate["reasons"])
    candidate_id = stable_id("conf", event_id, reason)
    stamp = now_ms()
    conn.execute(
        """
        INSERT INTO confirmation_candidates (
            candidate_id, event_id, reason, prompt, status, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
        ON CONFLICT(candidate_id) DO UPDATE SET
            reason = excluded.reason,
            prompt = excluded.prompt,
            updated_at_ms = excluded.updated_at_ms
        """,
        (candidate_id, event_id, reason, prompt, stamp, stamp),
    )
    return candidate_id


def base_atom(
    envelope: dict[str, Any],
    *,
    subject_scope: str,
    subject_id: str,
    atom_type: str,
    text: str,
    confidence: float,
    importance: float,
    sensitivity: str = "normal",
    status: str = "active",
    valid_until_ms: int | None = None,
) -> dict[str, Any]:
    event_id = str(envelope["event_id"])
    atom = {
        "memory_id": stable_id("mem", event_id, subject_scope, subject_id, atom_type, text),
        "subject_scope": subject_scope,
        "subject_id": subject_id,
        "type": atom_type,
        "text": text,
        "confidence": clamp01(confidence, 0.6),
        "importance": clamp01(importance, 0.5),
        "valid_from_ms": int(envelope.get("ts_ms") or now_ms()),
        "valid_until_ms": valid_until_ms,
        "sensitivity": sensitivity,
        "evidence_event_ids": [event_id],
        "source_turn_id": str(envelope.get("turn_id", "")),
        "status": status,
    }
    return atom


def deterministic_extract_atoms(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    text = normalize_text(get_path(envelope, "utterance.text", ""))
    subject_id = str(get_path(envelope, "subject.person_id", "") or "unknown")
    ts_ms = int(envelope.get("ts_ms") or now_ms())
    atoms: list[dict[str, Any]] = []
    sensitivity = "sensitive" if is_sensitive_text(text) else "normal"

    def add(atom_type: str, atom_text: str, confidence: float, importance: float) -> None:
        atoms.append(
            base_atom(
                envelope,
                subject_scope="person",
                subject_id=subject_id,
                atom_type=atom_type,
                text=atom_text,
                confidence=confidence,
                importance=importance,
                sensitivity=sensitivity,
            )
        )

    if text:
        if any(hint in text for hint in ("叫我", "我叫", "我是")):
            add("alias", text, 0.72, 0.62)

        if any(hint in text for hint in ("我喜欢", "我不喜欢", "我爱", "我讨厌", "不要给我", "别给我")):
            add("preference", text, 0.76, 0.68)

        commitment_hints = ("提醒", "记得", "别忘", "不要忘", "帮我", "待会", "明天", "后天", "下午", "上午", "点")
        if any(hint in text for hint in commitment_hints):
            atom_type = "commitment" if any(hint in text for hint in ("提醒", "记得", "别忘", "不要忘", "帮我")) else "plan"
            add(atom_type, text, 0.74, 0.82)

        if any(hint in text for hint in ("我现在", "我今天", "这会儿", "刚刚", "正在", "有点", "不舒服", "饿", "困", "累")):
            valid_until = ts_ms + 6 * 60 * 60 * 1000
            atoms.append(
                base_atom(
                    envelope,
                    subject_scope="person",
                    subject_id=subject_id,
                    atom_type="temporary_state",
                    text=text,
                    confidence=0.68,
                    importance=0.55,
                    sensitivity=sensitivity,
                    valid_until_ms=valid_until,
                )
            )

        if any(hint in text for hint in ("以后家里", "家里规则", "不要让", "不能让", "孩子不能", "机器人不要")):
            atoms.append(
                base_atom(
                    envelope,
                    subject_scope="household",
                    subject_id=str(envelope.get("household_id", "local_household")),
                    atom_type="rule",
                    text=text,
                    confidence=0.66,
                    importance=0.78,
                    sensitivity=sensitivity,
                )
            )

        if not atoms and "我" in text and len(compact_text(text)) >= 4:
            add("fact", text, 0.56, 0.42)

    for command in get_path(envelope, "robot_interaction.commands", []) or []:
        if not isinstance(command, dict):
            continue
        command_name = str(command.get("command") or command.get("type") or "command")
        ok = command.get("ok")
        result = command.get("message") or command.get("summary") or command.get("task_state") or ""
        atom_text = f"Robot command {command_name} result: {result}".strip()
        if ok is not None:
            atom_text += f" (ok={bool(ok)})"
        atoms.append(
            base_atom(
                envelope,
                subject_scope="robot",
                subject_id=str(envelope.get("robot_id", "local_robot")),
                atom_type="action_experience",
                text=atom_text,
                confidence=0.86,
                importance=0.5,
            )
        )
    return atoms


def build_extractor_messages(envelope: dict[str, Any]) -> list[dict[str, str]]:
    schema_hint = {
        "atoms": [
            {
                "subject_scope": "person|household|group|unknown|robot",
                "subject_id": "string",
                "type": "fact|preference|temporary_state|plan|commitment|alias|rule|action_experience",
                "text": "short evidence-grounded memory in Chinese if input is Chinese",
                "confidence": 0.0,
                "importance": 0.0,
                "valid_from_ms": int(envelope.get("ts_ms") or 0),
                "valid_until_ms": None,
                "sensitivity": "normal|sensitive",
                "status": "active|uncertain",
            }
        ]
    }
    system = (
        "/no_think\n"
        "You extract durable robot memory atoms from one JSON envelope. "
        "Use only the evidence in the envelope. Do not infer identity. "
        "Do not create personal memories for anyone except envelope.subject.person_id. "
        "Return valid JSON only. No markdown. No reasoning text."
    )
    user = {
        "task": "extract_memory_atoms",
        "output_schema": schema_hint,
        "rules": [
            "Prefer no atom over speculative atom.",
            "Temporary states need valid_until_ms when obvious.",
            "Sensitive data is allowed only when the utterance explicitly asks the robot to remember it.",
            "Robot execution outcomes should use subject_scope=robot and type=action_experience.",
        ],
        "envelope": envelope,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "/no_think\n" + json_dumps(user)},
    ]


def call_llm_json(
    config: ModelConfig,
    *,
    task: str,
    event_id: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response_text = ""
    error = ""
    ok = False
    try:
        request = urllib.request.Request(
            config.endpoint,
            data=json_dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        response_text = get_path(body, "choices.0.message.content", "")
        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON response must be an object")
        ok = True
        return parsed
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        error = str(exc)
        repaired = try_repair_llm_json(config, response_text, task, event_id, conn)
        if repaired is not None:
            ok = True
            return repaired
        raise RuntimeError(f"LLM {task} failed: {error}") from exc
    finally:
        if conn is not None:
            log_model_call(
                conn,
                event_id=event_id,
                task=task,
                endpoint=config.endpoint,
                model=config.model,
                ok=ok,
                error=error,
                request_json=payload,
                response_text=response_text,
            )


def try_repair_llm_json(
    config: ModelConfig,
    response_text: str,
    task: str,
    event_id: str,
    conn: sqlite3.Connection | None,
) -> dict[str, Any] | None:
    if not response_text.strip():
        return None
    messages = [
        {
            "role": "system",
            "content": "Repair the user content into valid JSON. Return JSON only.",
        },
        {"role": "user", "content": response_text[:6000]},
    ]
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 768,
        "response_format": {"type": "json_object"},
    }
    repaired_text = ""
    try:
        request = urllib.request.Request(
            config.endpoint,
            data=json_dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        repaired_text = get_path(body, "choices.0.message.content", "")
        repaired = json.loads(repaired_text)
        if isinstance(repaired, dict):
            return repaired
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    finally:
        if conn is not None and repaired_text:
            log_model_call(
                conn,
                event_id=event_id,
                task=f"{task}.repair",
                endpoint=config.endpoint,
                model=config.model,
                ok=False,
                error="repair_attempt",
                request_json=payload,
                response_text=repaired_text,
            )
    return None


def log_model_call(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    task: str,
    endpoint: str,
    model: str,
    ok: bool,
    error: str,
    request_json: Any,
    response_text: str,
) -> None:
    stamp = now_ms()
    conn.execute(
        """
        INSERT INTO model_call_logs (
            call_id, event_id, task, endpoint, model, ok, error,
            request_json, response_text, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stable_id("llm", event_id, task, stamp),
            event_id,
            task,
            endpoint,
            model,
            int(ok),
            error,
            json_dumps(request_json),
            response_text,
            stamp,
        ),
    )


def validate_atom(
    raw_atom: dict[str, Any], envelope: dict[str, Any], index: int
) -> dict[str, Any] | None:
    text = normalize_text(raw_atom.get("text"))
    if not text or len(compact_text(text)) < 2:
        return None
    subject_scope = str(raw_atom.get("subject_scope") or "person")
    if subject_scope not in SUBJECT_SCOPES:
        subject_scope = "person"
    if subject_scope == "person":
        subject_id = str(get_path(envelope, "subject.person_id", "") or "unknown")
    elif subject_scope == "robot":
        subject_id = str(envelope.get("robot_id", "local_robot"))
    elif subject_scope == "household":
        subject_id = str(envelope.get("household_id", "local_household"))
    else:
        subject_id = str(raw_atom.get("subject_id") or subject_scope)

    atom_type = str(raw_atom.get("type") or "fact")
    if atom_type not in ATOM_TYPES:
        return None
    sensitivity = str(raw_atom.get("sensitivity") or "normal")
    if sensitivity not in SENSITIVITY_VALUES:
        sensitivity = "normal"
    status = str(raw_atom.get("status") or "active")
    if status not in ATOM_STATUSES:
        status = "uncertain"
    valid_until_ms = raw_atom.get("valid_until_ms")
    if valid_until_ms is not None:
        try:
            valid_until_ms = int(valid_until_ms)
        except (TypeError, ValueError):
            valid_until_ms = None

    event_id = str(envelope["event_id"])
    atom = {
        "memory_id": stable_id(
            "mem", event_id, subject_scope, subject_id, atom_type, text, index
        ),
        "subject_scope": subject_scope,
        "subject_id": subject_id,
        "type": atom_type,
        "text": text[:600],
        "confidence": clamp01(raw_atom.get("confidence"), 0.6),
        "importance": clamp01(raw_atom.get("importance"), 0.5),
        "valid_from_ms": int(raw_atom.get("valid_from_ms") or envelope.get("ts_ms") or now_ms()),
        "valid_until_ms": valid_until_ms,
        "sensitivity": sensitivity,
        "evidence_event_ids": [event_id],
        "source_turn_id": str(envelope.get("turn_id", "")),
        "status": status,
    }
    return atom


def extract_atoms(
    conn: sqlite3.Connection,
    envelope: dict[str, Any],
    config: ModelConfig,
) -> tuple[list[dict[str, Any]], str]:
    if config.no_llm:
        return deterministic_extract_atoms(envelope), "rules"

    messages = build_extractor_messages(envelope)
    try:
        parsed = call_llm_json(
            config,
            task="extractor",
            event_id=str(envelope["event_id"]),
            messages=messages,
            temperature=0,
            max_tokens=768,
            conn=conn,
        )
    except RuntimeError:
        if not config.fallback_on_error:
            raise
        return deterministic_extract_atoms(envelope), "rules_after_llm_error"

    raw_atoms = parsed.get("atoms", [])
    if not isinstance(raw_atoms, list):
        raise ValueError("extractor output must contain atoms list")
    atoms: list[dict[str, Any]] = []
    for index, raw_atom in enumerate(raw_atoms):
        if not isinstance(raw_atom, dict):
            continue
        atom = validate_atom(raw_atom, envelope, index)
        if atom is not None:
            atoms.append(atom)
    return atoms, "llm"


def insert_memory_atoms(
    conn: sqlite3.Connection, envelope: dict[str, Any], atoms: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    stamp = now_ms()
    inserted: list[dict[str, Any]] = []
    for atom in atoms:
        memory_id = str(atom.get("memory_id") or stable_id("mem", envelope["event_id"], atom))
        atom["memory_id"] = memory_id
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_atoms (
                memory_id, event_id, subject_scope, subject_id, type, text,
                confidence, importance, valid_from_ms, valid_until_ms, sensitivity,
                evidence_event_ids, source_turn_id, status, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                str(envelope["event_id"]),
                atom["subject_scope"],
                atom["subject_id"],
                atom["type"],
                atom["text"],
                float(atom["confidence"]),
                float(atom["importance"]),
                int(atom["valid_from_ms"]),
                atom.get("valid_until_ms"),
                atom["sensitivity"],
                json_dumps(atom["evidence_event_ids"]),
                atom["source_turn_id"],
                atom["status"],
                stamp,
                stamp,
            ),
        )
        if conn.total_changes and conn.execute("SELECT changes()").fetchone()[0] > 0:
            inserted.append(atom)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_atoms_fts(memory_id, text) VALUES (?, ?)",
                    (memory_id, atom["text"]),
                )
            except sqlite3.OperationalError:
                pass
    return inserted


def empty_profile(person_id: str, display_name: str) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "display_name": display_name,
        "aliases": [],
        "stable_facts": [],
        "preferences": [],
        "communication_style": [],
        "recent_context": [],
        "boundaries": [],
        "uncertain": [],
        "evidence_memory_ids": [],
    }


def fetch_profile(
    conn: sqlite3.Connection, person_id: str, display_name: str = ""
) -> dict[str, Any]:
    row = conn.execute(
        "SELECT profile_json, display_name FROM person_profiles WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    if row is None:
        return empty_profile(person_id, display_name)
    profile = json.loads(row["profile_json"])
    if display_name and not profile.get("display_name"):
        profile["display_name"] = display_name
    if not profile.get("display_name"):
        profile["display_name"] = row["display_name"]
    return profile


def profile_category_for_atom(atom: dict[str, Any]) -> str:
    if atom["status"] == "uncertain" or atom["confidence"] < 0.55:
        return "uncertain"
    atom_type = atom["type"]
    if atom_type == "alias":
        return "aliases"
    if atom_type == "fact":
        return "stable_facts"
    if atom_type == "preference":
        return "preferences"
    if atom_type in {"rule"}:
        return "boundaries"
    return "recent_context"


def apply_profile_atoms(
    conn: sqlite3.Connection,
    person_id: str,
    display_name: str,
    atoms: list[dict[str, Any]],
) -> dict[str, Any] | None:
    personal_atoms = [
        atom
        for atom in atoms
        if atom.get("subject_scope") == "person"
        and atom.get("subject_id") == person_id
        and atom.get("status") in {"active", "uncertain"}
    ]
    if not personal_atoms:
        return None

    profile = fetch_profile(conn, person_id, display_name)
    evidence_ids = set(profile.get("evidence_memory_ids", []))
    changed = False
    for atom in personal_atoms:
        memory_id = atom["memory_id"]
        if memory_id in evidence_ids:
            continue
        category = profile_category_for_atom(atom)
        entry = {
            "memory_id": memory_id,
            "text": atom["text"],
            "confidence": atom["confidence"],
            "importance": atom["importance"],
            "updated_at_ms": now_ms(),
        }
        profile.setdefault(category, []).append(entry)
        profile.setdefault("evidence_memory_ids", []).append(memory_id)
        evidence_ids.add(memory_id)
        changed = True

    if not changed:
        return profile

    stamp = now_ms()
    conn.execute(
        """
        INSERT INTO person_profiles (
            person_id, display_name, profile_json, version, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, 1, ?, ?)
        ON CONFLICT(person_id) DO UPDATE SET
            display_name = excluded.display_name,
            profile_json = excluded.profile_json,
            version = version + 1,
            updated_at_ms = excluded.updated_at_ms
        """,
        (person_id, display_name, json_dumps(profile), stamp, stamp),
    )
    return profile


def build_profile_patch_from_atoms(person_id: str, atoms: list[dict[str, Any]]) -> dict[str, Any]:
    patch = empty_profile(person_id, "")
    for atom in atoms:
        category = profile_category_for_atom(atom)
        patch.setdefault(category, []).append(atom["text"])
        patch["evidence_memory_ids"].append(atom["memory_id"])
    return patch


def merge_session_profile(
    conn: sqlite3.Connection,
    session_id: str,
    person_id: str,
    config: ModelConfig,
) -> dict[str, Any]:
    params: list[Any] = [person_id]
    sql = """
        SELECT a.*
        FROM memory_atoms a
        JOIN raw_memory_events e ON e.event_id = a.event_id
        WHERE a.subject_scope = 'person'
          AND a.subject_id = ?
          AND a.status IN ('active', 'uncertain')
    """
    if session_id:
        sql += " AND e.session_id = ?"
        params.append(session_id)
    sql += " ORDER BY a.updated_at_ms ASC"
    rows = conn.execute(sql, params).fetchall()
    atoms = [row_to_atom(row) for row in rows]
    if config.no_llm or not atoms:
        patch = build_profile_patch_from_atoms(person_id, atoms)
    else:
        messages = build_profile_merge_messages(session_id, person_id, atoms)
        try:
            patch = call_llm_json(
                config,
                task="profile_merge",
                event_id=session_id or f"person:{person_id}",
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                conn=conn,
            )
        except RuntimeError:
            if not config.fallback_on_error:
                raise
            patch = build_profile_patch_from_atoms(person_id, atoms)
    display_name = ""
    raw = conn.execute(
        "SELECT raw_json FROM raw_memory_events WHERE subject_id = ? ORDER BY ts_ms DESC LIMIT 1",
        (person_id,),
    ).fetchone()
    if raw is not None:
        display_name = normalize_text(get_path(json.loads(raw["raw_json"]), "subject.display_name", ""))
    profile_atoms = [
        atom for atom in atoms if atom["memory_id"] in set(patch.get("evidence_memory_ids", []))
    ]
    profile = apply_profile_atoms(conn, person_id, display_name, profile_atoms) or fetch_profile(
        conn, person_id, display_name
    )
    profile["last_profile_patch"] = patch
    stamp = now_ms()
    conn.execute(
        """
        UPDATE person_profiles
        SET profile_json = ?, updated_at_ms = ?, version = version + 1
        WHERE person_id = ?
        """,
        (json_dumps(profile), stamp, person_id),
    )
    return profile


def build_profile_merge_messages(
    session_id: str, person_id: str, atoms: list[dict[str, Any]]
) -> list[dict[str, str]]:
    system = (
        "Merge memory atoms into a person profile patch. "
        "Do not overwrite conflicts; place uncertain items in uncertain. "
        "Return JSON only with keys person_id, aliases, stable_facts, preferences, "
        "communication_style, recent_context, boundaries, uncertain, evidence_memory_ids."
    )
    user = {
        "task": "profile_merge",
        "session_id": session_id,
        "person_id": person_id,
        "atoms": atoms,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json_dumps(user)},
    ]


def row_to_atom(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "memory_id": row["memory_id"],
        "event_id": row["event_id"],
        "subject_scope": row["subject_scope"],
        "subject_id": row["subject_id"],
        "type": row["type"],
        "text": row["text"],
        "confidence": row["confidence"],
        "importance": row["importance"],
        "valid_from_ms": row["valid_from_ms"],
        "valid_until_ms": row["valid_until_ms"],
        "sensitivity": row["sensitivity"],
        "evidence_event_ids": json.loads(row["evidence_event_ids"]),
        "source_turn_id": row["source_turn_id"],
        "status": row["status"],
        "created_at_ms": row["created_at_ms"],
        "updated_at_ms": row["updated_at_ms"],
    }


def row_to_raw_event(row: sqlite3.Row) -> dict[str, Any]:
    raw = json.loads(row["raw_json"])
    return {
        "event_id": row["event_id"],
        "session_id": row["session_id"],
        "turn_id": row["turn_id"],
        "ts_ms": row["ts_ms"],
        "subject_id": row["subject_id"],
        "speaker_state": row["speaker_state"],
        "allow_memory_write": bool(row["allow_memory_write"]),
        "allow_profile_write": bool(row["allow_profile_write"]),
        "need_clarification": bool(row["need_clarification"]),
        "profile_write_reason": row["profile_write_reason"],
        "utterance_text": normalize_text(get_path(raw, "utterance.text", "")),
        "scene_summary": normalize_text(get_path(raw, "slow_reaction.scene.summary", "")),
        "robot_reply": normalize_text(get_path(raw, "robot_interaction.reply_text", "")),
        "source_refs": raw.get("source_refs", {}),
        "updated_at_ms": row["updated_at_ms"],
    }


def row_to_short_term(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "state_id": row["state_id"],
        "event_id": row["event_id"],
        "session_id": row["session_id"],
        "turn_id": row["turn_id"],
        "subject_id": row["subject_id"],
        "state_type": row["state_type"],
        "summary": row["summary"],
        "payload": json.loads(row["payload_json"]),
        "created_at_ms": row["created_at_ms"],
        "updated_at_ms": row["updated_at_ms"],
    }


def row_to_confirmation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "event_id": row["event_id"],
        "reason": row["reason"],
        "prompt": row["prompt"],
        "status": row["status"],
        "created_at_ms": row["created_at_ms"],
        "updated_at_ms": row["updated_at_ms"],
    }


def ingest_envelope(
    conn: sqlite3.Connection,
    envelope: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    envelope = normalize_envelope(envelope)
    existed = upsert_raw_event(conn, envelope)
    gate = evaluate_write_gate(envelope)
    short_term_ids = insert_short_term_entries(conn, build_short_term_entries(envelope, gate))
    confirmation_id = insert_confirmation_candidate(conn, envelope, gate)
    atoms: list[dict[str, Any]] = []
    inserted_atoms: list[dict[str, Any]] = []
    extractor = "none"
    profile = None

    if gate["allowed"]:
        atoms, extractor = extract_atoms(conn, envelope, config)
        inserted_atoms = insert_memory_atoms(conn, envelope, atoms)
        profile = apply_profile_atoms(
            conn,
            str(get_path(envelope, "subject.person_id", "")),
            normalize_text(get_path(envelope, "subject.display_name", "")),
            inserted_atoms,
        )

    conn.commit()
    return {
        "ok": True,
        "event_id": envelope["event_id"],
        "session_id": envelope["session_id"],
        "turn_id": envelope["turn_id"],
        "raw_event_existed": existed,
        "personal_write_allowed": gate["allowed"],
        "write_gate": gate,
        "short_term_state_ids": short_term_ids,
        "confirmation_candidate_id": confirmation_id,
        "extractor": extractor,
        "atom_count": len(atoms),
        "inserted_atom_count": len(inserted_atoms),
        "memory_ids": [atom["memory_id"] for atom in inserted_atoms],
        "inserted_atoms": inserted_atoms,
        "profile_updated": profile is not None,
    }


def query_context_pack(
    conn: sqlite3.Connection,
    *,
    person_id: str,
    text: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    params: list[Any] = ["person", person_id]
    sql = (
        "SELECT * FROM memory_atoms "
        "WHERE subject_scope = ? AND subject_id = ? AND status IN ('active', 'uncertain')"
    )
    if text:
        try:
            fts_rows = conn.execute(
                """
                SELECT a.*
                FROM memory_atoms_fts f
                JOIN memory_atoms a ON a.memory_id = f.memory_id
                WHERE memory_atoms_fts MATCH ?
                  AND a.subject_scope = 'person'
                  AND a.subject_id = ?
                  AND a.status IN ('active', 'uncertain')
                ORDER BY rank
                LIMIT ?
                """,
                (text, person_id, limit),
            ).fetchall()
            atoms = [row_to_atom(row) for row in fts_rows]
            return {
                "person_id": person_id,
                "profile": fetch_profile(conn, person_id),
                "atoms": atoms,
                "retrieval": {"mode": "fts5", "query": text, "limit": limit},
            }
        except sqlite3.OperationalError:
            sql += " AND text LIKE ?"
            params.append(f"%{text}%")
    sql += " ORDER BY importance DESC, updated_at_ms DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return {
        "person_id": person_id,
        "profile": fetch_profile(conn, person_id),
        "atoms": [row_to_atom(row) for row in rows],
        "retrieval": {"mode": "sqlite_like" if text else "sqlite_filter", "query": text, "limit": limit},
    }


def query_memory_tiers(
    conn: sqlite3.Connection,
    *,
    person_id: str,
    session_id: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    raw_params: list[Any] = [person_id]
    raw_sql = "SELECT * FROM raw_memory_events WHERE subject_id = ?"
    if session_id:
        raw_sql += " AND session_id = ?"
        raw_params.append(session_id)
    raw_sql += " ORDER BY ts_ms DESC LIMIT ?"
    raw_params.append(limit)
    raw_rows = conn.execute(raw_sql, raw_params).fetchall()
    raw_events = [row_to_raw_event(row) for row in raw_rows]

    short_params: list[Any] = [person_id]
    short_sql = "SELECT * FROM short_term_state WHERE subject_id = ?"
    if session_id:
        short_sql += " AND session_id = ?"
        short_params.append(session_id)
    short_sql += " ORDER BY updated_at_ms DESC LIMIT ?"
    short_params.append(limit)
    short_term = [row_to_short_term(row) for row in conn.execute(short_sql, short_params)]

    atom_params: list[Any] = ["person", person_id]
    atom_sql = (
        "SELECT * FROM memory_atoms "
        "WHERE subject_scope = ? AND subject_id = ? AND status IN ('active', 'uncertain')"
    )
    if session_id:
        atom_sql += " AND event_id IN (SELECT event_id FROM raw_memory_events WHERE session_id = ?)"
        atom_params.append(session_id)
    atom_sql += " ORDER BY importance DESC, updated_at_ms DESC LIMIT ?"
    atom_params.append(limit)
    atoms = [row_to_atom(row) for row in conn.execute(atom_sql, atom_params)]

    confirmation_params: list[Any] = [person_id]
    confirmation_sql = (
        "SELECT c.* FROM confirmation_candidates c "
        "JOIN raw_memory_events e ON e.event_id = c.event_id "
        "WHERE e.subject_id = ?"
    )
    if session_id:
        confirmation_sql += " AND e.session_id = ?"
        confirmation_params.append(session_id)
    confirmation_sql += " ORDER BY c.updated_at_ms DESC LIMIT ?"
    confirmation_params.append(limit)
    confirmations = [
        row_to_confirmation(row)
        for row in conn.execute(confirmation_sql, confirmation_params)
    ]

    profile = fetch_profile(conn, person_id)
    stats = {
        "l0_raw_events": len(raw_events),
        "short_term_items": len(short_term),
        "mid_term_atoms": len(atoms),
        "long_term_profile_items": len(profile.get("evidence_memory_ids", [])),
        "confirmation_candidates": len(confirmations),
    }
    return {
        "ok": True,
        "person_id": person_id,
        "session_id": session_id,
        "tiers": {
            "l0_raw_ledger": raw_events,
            "short_term": short_term,
            "mid_term_atoms": atoms,
            "long_term_profile": profile,
            "confirmation_candidates": confirmations,
        },
        "stats": stats,
    }


def display_name_for_person(conn: sqlite3.Connection, person_id: str) -> str:
    row = conn.execute(
        "SELECT display_name FROM person_profiles WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    if row is not None and row["display_name"]:
        return str(row["display_name"])
    row = conn.execute(
        """
        SELECT raw_json FROM raw_memory_events
        WHERE subject_id = ?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (person_id,),
    ).fetchone()
    if row is not None:
        display_name = normalize_text(
            get_path(json.loads(row["raw_json"]), "subject.display_name", "")
        )
        if display_name:
            return display_name
    return person_id


def list_people(conn: sqlite3.Connection, limit: int = 100) -> dict[str, Any]:
    person_ids: set[str] = set()
    sources = [
        ("person_profiles", "person_id", ""),
        ("raw_memory_events", "subject_id", "WHERE subject_id != ''"),
        ("short_term_state", "subject_id", "WHERE subject_id != ''"),
        (
            "memory_atoms",
            "subject_id",
            "WHERE subject_scope = 'person' AND subject_id != ''",
        ),
    ]
    for table, column, where in sources:
        rows = conn.execute(
            f"SELECT DISTINCT {column} AS person_id FROM {table} {where}"
        ).fetchall()
        for row in rows:
            person_id = str(row["person_id"])
            if person_id and person_id not in {"unknown", "group", "robot"}:
                person_ids.add(person_id)

    people: list[dict[str, Any]] = []
    for person_id in person_ids:
        profile = fetch_profile(conn, person_id)
        short_count = conn.execute(
            "SELECT COUNT(*) FROM short_term_state WHERE subject_id = ?",
            (person_id,),
        ).fetchone()[0]
        atom_count = conn.execute(
            """
            SELECT COUNT(*) FROM memory_atoms
            WHERE subject_scope = 'person'
              AND subject_id = ?
              AND status IN ('active', 'uncertain')
            """,
            (person_id,),
        ).fetchone()[0]
        raw_count = conn.execute(
            "SELECT COUNT(*) FROM raw_memory_events WHERE subject_id = ?",
            (person_id,),
        ).fetchone()[0]
        confirmation_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM confirmation_candidates c
            JOIN raw_memory_events e ON e.event_id = c.event_id
            WHERE e.subject_id = ?
            """,
            (person_id,),
        ).fetchone()[0]
        last_seen = conn.execute(
            "SELECT MAX(ts_ms) AS last_seen FROM raw_memory_events WHERE subject_id = ?",
            (person_id,),
        ).fetchone()["last_seen"]
        people.append(
            {
                "person_id": person_id,
                "display_name": display_name_for_person(conn, person_id),
                "counts": {
                    "short_term": short_count,
                    "mid_term": atom_count,
                    "long_term": len(profile.get("evidence_memory_ids", [])),
                    "raw_events": raw_count,
                    "confirmation_candidates": confirmation_count,
                },
                "last_seen_ms": last_seen or 0,
            }
        )

    people.sort(
        key=lambda item: (item["last_seen_ms"], item["counts"]["mid_term"]),
        reverse=True,
    )
    return {"ok": True, "people": people[:limit]}


def map_speaker_state(speaker_turn: dict[str, Any]) -> str:
    state = str(speaker_turn.get("speaker_state") or get_path(speaker_turn, "speaker_source.state", ""))
    evidence = speaker_turn.get("evidence", {}) if isinstance(speaker_turn, dict) else {}
    if state in SPEAKER_STATES:
        return state
    if state in {"overlap", "mixed", "overlap_speech"} or as_bool(evidence.get("overlap")):
        return "overlap_speech"
    if state in {"confirmed_visible", "known_visible", "visible_speaker"}:
        return "known_visible_speaker"
    if state in {"confirmed_offscreen", "known_offscreen", "offscreen_speaker"}:
        return "known_offscreen_speaker"
    source_location = str(get_path(speaker_turn, "speaker_source.location", ""))
    selection_reason = str(get_path(speaker_turn, "speaker_source.selection_reason", ""))
    if source_location == "offscreen" or "offscreen" in selection_reason:
        return "known_offscreen_speaker"
    return "uncertain"


def load_scheduler_event(
    path: pathlib.Path, session_id: str = "", turn_id: str = ""
) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix == ".ndjson":
        selected: dict[str, Any] = {}
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if session_id and item.get("session_id") != session_id:
                    continue
                if turn_id and item.get("turn_id") not in {turn_id, f"voice_{turn_id}"}:
                    continue
                selected = item
        return selected
    return load_json(path)


def compose_from_proactive_os(
    *,
    voice_state: dict[str, Any],
    slow_scene: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    scheduler_event: dict[str, Any] | None = None,
    proactive_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turn = voice_state.get("last_turn", voice_state)
    if not isinstance(turn, dict):
        raise ValueError("voice_state must contain last_turn object or be a turn object")

    speaker_turn = turn.get("speaker_turn", {}) or {}
    audio_event = turn.get("audio_event", {}) or {}
    session_id = str(turn.get("session_id") or speaker_turn.get("session_id") or "")
    turn_id = str(turn.get("turn_id") or speaker_turn.get("turn_id") or audio_event.get("turn_id") or "")
    ts_ms = int(
        audio_event.get("ts_ms")
        or audio_event.get("end_ms")
        or speaker_turn.get("updated_at_ms")
        or turn.get("ts_ms")
        or now_ms()
    )
    person_id = str(turn.get("person_id") or speaker_turn.get("person_id") or "")
    display_name = str(turn.get("display_name") or speaker_turn.get("display_name") or "")
    scheduler_event = scheduler_event or {}
    slow_scene = slow_scene or {}
    environment = environment or {}
    proactive_state = proactive_state or {}

    commands = scheduler_event.get("command_results", [])
    if not isinstance(commands, list):
        commands = []
    reply_text = ""
    for command in commands:
        if not isinstance(command, dict):
            continue
        request = get_path(command, "summary.request", {}) or {}
        if isinstance(request, dict) and request.get("text"):
            reply_text = str(request["text"])
            break

    snapshot_refs: list[str] = []
    scene_frame_path = get_path(slow_scene, "control.scene_frame_path")
    if scene_frame_path:
        snapshot_refs.append(str(scene_frame_path))
    for candidate in get_path(speaker_turn, "visual.candidates", []) or []:
        if isinstance(candidate, dict) and candidate.get("face_crop_ref"):
            snapshot_refs.append(str(candidate["face_crop_ref"]))

    audio_refs: list[str] = []
    pcm_ref = audio_event.get("pcm_ref") or get_path(audio_event, "audio_activity.feature_contract.pcm_ref")
    if pcm_ref:
        audio_refs.append(str(pcm_ref))

    policy = speaker_turn.get("policy", {}) if isinstance(speaker_turn.get("policy"), dict) else {}
    allow_profile_write = as_bool(speaker_turn.get("allow_profile_write"))
    allow_memory_write = as_bool(policy.get("allow_memory_write"))
    profile_write_reason = str(speaker_turn.get("profile_write_reason") or "")

    envelope = {
        "schema": INPUT_SCHEMA,
        "event_id": stable_id("mem_in", session_id, turn_id, ts_ms),
        "session_id": session_id,
        "turn_id": turn_id,
        "household_id": "local_household",
        "robot_id": "local_robot",
        "ts_ms": ts_ms,
        "subject": {
            "person_id": person_id,
            "display_name": display_name,
            "speaker_state": map_speaker_state(speaker_turn),
            "identity_context": speaker_turn.get("identity_context", {}),
            "speaker_source": speaker_turn.get("speaker_source", {}),
            "speaker_probabilities": speaker_turn.get("speaker_probabilities", []),
        },
        "utterance": {
            "text": str(turn.get("text") or speaker_turn.get("text") or audio_event.get("asr_text") or ""),
            "language": str(audio_event.get("language") or "zh"),
            "start_ms": int(audio_event.get("start_ms") or get_path(speaker_turn, "audio.start_ms", 0) or 0),
            "end_ms": int(audio_event.get("end_ms") or get_path(speaker_turn, "audio.end_ms", 0) or 0),
            "asr_confidence": float(audio_event.get("asr_confidence") or speaker_turn.get("confidence") or 0.0),
            "segments": get_path(audio_event, "speaker_attributed_transcript.segments", []),
        },
        "fast_reaction": {
            "speaker_turn": speaker_turn,
            "audio_event": audio_event,
            "visual": speaker_turn.get("visual", turn.get("visual", {})) or {},
            "nlu": turn.get("nlu", {}) or {},
            "policy": {
                "allow_memory_write": allow_memory_write,
                "allow_profile_write": allow_profile_write,
                "need_clarification": as_bool(policy.get("need_clarification")),
                "profile_write_reason": profile_write_reason,
            },
        },
        "slow_reaction": {
            "scene": {
                "summary": str(slow_scene.get("response_text") or ""),
                "activity": "",
                "objects": [],
                "room": "",
                "people": [],
            },
            "environment": environment,
            "scene_model": str(slow_scene.get("model") or get_path(slow_scene, "control.llm.model_name", "")),
        },
        "robot_interaction": {
            "reply_text": reply_text,
            "commands": commands,
            "scheduler_result": scheduler_event,
            "proactive_candidate": proactive_state.get("last_candidate", {}),
            "execution_result": proactive_state.get("last_execution", {}),
        },
        "source_refs": {
            "raw_event_ids": [
                item
                for item in [
                    turn.get("type"),
                    speaker_turn.get("type"),
                    slow_scene.get("type"),
                    scheduler_event.get("type"),
                ]
                if item
            ],
            "snapshot_refs": snapshot_refs,
            "audio_refs": audio_refs,
        },
    }
    return normalize_envelope(envelope)


class MemoryHTTPHandler(http.server.BaseHTTPRequestHandler):
    db_path: pathlib.Path = DEFAULT_DB_PATH
    model_config: ModelConfig = ModelConfig()

    def _send_json(self, status: int, payload: Any) -> None:
        body = json_dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: pathlib.Path) -> None:
        try:
            root = UI_ROOT.resolve()
            target = path.resolve()
            if root not in target.parents and target != root:
                self._send_json(403, {"ok": False, "error": "forbidden"})
                return
            if not target.exists() or not target.is_file():
                self._send_json(404, {"ok": False, "error": "not_found"})
                return
            body = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def _redirect(self, target: str) -> None:
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_OPTIONS(self) -> None:
        self._send_json(204, {})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            with connect_db(self.db_path) as conn:
                init_db(conn)
                if self.path == "/memory/events":
                    result = ingest_envelope(conn, payload, self.model_config)
                elif self.path == "/memory/context-pack":
                    result = query_context_pack(
                        conn,
                        person_id=str(payload.get("person_id") or ""),
                        text=str(payload.get("text") or ""),
                        limit=int(payload.get("limit") or 20),
                    )
                elif self.path == "/memory/tier-dashboard":
                    result = query_memory_tiers(
                        conn,
                        person_id=str(payload.get("person_id") or ""),
                        session_id=str(payload.get("session_id") or ""),
                        limit=int(payload.get("limit") or 20),
                    )
                elif self.path == "/memory/profile-merge":
                    result = {
                        "ok": True,
                        "profile": merge_session_profile(
                            conn,
                            session_id=str(payload.get("session_id") or ""),
                            person_id=str(payload.get("person_id") or ""),
                            config=self.model_config,
                        ),
                    }
                    conn.commit()
                elif self.path == "/memory/people":
                    result = list_people(conn, limit=int(payload.get("limit") or 100))
                else:
                    self._send_json(404, {"ok": False, "error": "not_found"})
                    return
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001 - this is a tiny stdlib HTTP boundary.
            self._send_json(400, {"ok": False, "error": str(exc)})

    def do_GET(self) -> None:
        try:
            path = self.path.split("?", 1)[0]
            if path == "/memory/ui":
                self._redirect("/memory/ui/")
                return
            if path in {"/", "/memory/ui/"}:
                self._send_static(UI_ROOT / "memory.html")
                return
            if path.startswith("/memory/ui/"):
                relative = path.removeprefix("/memory/ui/")
                self._send_static(UI_ROOT / relative)
                return
            if path == "/memory/config":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "db": str(self.db_path),
                        "endpoint": self.model_config.endpoint,
                        "model": self.model_config.model,
                        "default_model_path": str(DEFAULT_MODEL_PATH),
                        "no_llm": self.model_config.no_llm,
                        "fallback_on_llm_error": self.model_config.fallback_on_error,
                    },
                )
                return
            if path == "/memory/people":
                with connect_db(self.db_path) as conn:
                    init_db(conn)
                    result = list_people(conn)
                self._send_json(200, result)
                return
            prefix = "/memory/person/"
            suffix = "/tiers"
            if path.startswith(prefix) and path.endswith(suffix):
                person_id = path[len(prefix) : -len(suffix)]
                with connect_db(self.db_path) as conn:
                    init_db(conn)
                    result = query_memory_tiers(conn, person_id=person_id, limit=50)
                self._send_json(200, result)
                return
            prefix = "/memory/person/"
            suffix = "/dashboard"
            if not (path.startswith(prefix) and path.endswith(suffix)):
                self._send_json(404, {"ok": False, "error": "not_found"})
                return
            person_id = path[len(prefix) : -len(suffix)]
            with connect_db(self.db_path) as conn:
                init_db(conn)
                result = query_context_pack(conn, person_id=person_id, limit=50)
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("memory_service: " + fmt % args + "\n")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_LLM_TIMEOUT_S)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--fallback-on-llm-error", action="store_true")


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        endpoint=args.endpoint,
        model=args.model,
        no_llm=args.no_llm,
        timeout_s=args.timeout_s,
        fallback_on_error=args.fallback_on_llm_error,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="create or migrate the SQLite memory database")

    ingest = subparsers.add_parser("ingest", help="ingest one robot.memory.input.v1 envelope")
    ingest.add_argument("--input", required=True, help="input JSON path, or '-' for stdin")
    add_model_args(ingest)

    compose = subparsers.add_parser("compose", help="compose an envelope from ProactiveOS outputs")
    compose.add_argument("--voice-state", type=pathlib.Path, required=True)
    compose.add_argument("--slow-scene", type=pathlib.Path)
    compose.add_argument("--environment", type=pathlib.Path)
    compose.add_argument("--scheduler-event", type=pathlib.Path)
    compose.add_argument("--proactive-state", type=pathlib.Path)
    compose.add_argument("--output", type=pathlib.Path)

    query = subparsers.add_parser("query", help="query a person's memory context pack")
    query.add_argument("--person-id", required=True)
    query.add_argument("--text", default="")
    query.add_argument("--limit", type=int, default=20)

    merge = subparsers.add_parser("merge-session", help="merge session atoms into a person profile")
    merge.add_argument("--session-id", required=True)
    merge.add_argument("--person-id", required=True)
    add_model_args(merge)

    serve = subparsers.add_parser("serve", help="serve HTTP endpoints")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8095)
    add_model_args(serve)
    return parser


def command_init_db(args: argparse.Namespace) -> int:
    with connect_db(args.db) as conn:
        init_db(conn)
    print(json_dumps({"ok": True, "db": str(args.db)}, indent=2))
    return 0


def command_ingest(args: argparse.Namespace) -> int:
    envelope = load_json_or_stdin(args.input)
    with connect_db(args.db) as conn:
        init_db(conn)
        result = ingest_envelope(conn, envelope, model_config_from_args(args))
    print(json_dumps(result, indent=2))
    return 0


def command_compose(args: argparse.Namespace) -> int:
    voice_state = load_json(args.voice_state)
    slow_scene = load_json(args.slow_scene) if args.slow_scene else None
    environment = load_json(args.environment) if args.environment else None
    scheduler_event = None
    if args.scheduler_event:
        turn = voice_state.get("last_turn", voice_state)
        scheduler_event = load_scheduler_event(
            args.scheduler_event,
            session_id=str(turn.get("session_id") or ""),
            turn_id=str(turn.get("turn_id") or ""),
        )
    proactive_state = load_json(args.proactive_state) if args.proactive_state else None
    envelope = compose_from_proactive_os(
        voice_state=voice_state,
        slow_scene=slow_scene,
        environment=environment,
        scheduler_event=scheduler_event,
        proactive_state=proactive_state,
    )
    output = json_dumps(envelope, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


def command_query(args: argparse.Namespace) -> int:
    with connect_db(args.db) as conn:
        init_db(conn)
        result = query_context_pack(
            conn, person_id=args.person_id, text=args.text, limit=args.limit
        )
    print(json_dumps(result, indent=2))
    return 0


def command_merge_session(args: argparse.Namespace) -> int:
    with connect_db(args.db) as conn:
        init_db(conn)
        result = merge_session_profile(
            conn,
            session_id=args.session_id,
            person_id=args.person_id,
            config=model_config_from_args(args),
        )
        conn.commit()
    print(json_dumps({"ok": True, "profile": result}, indent=2))
    return 0


def command_serve(args: argparse.Namespace) -> int:
    with connect_db(args.db) as conn:
        init_db(conn)
    MemoryHTTPHandler.db_path = args.db
    MemoryHTTPHandler.model_config = model_config_from_args(args)
    server = http.server.ThreadingHTTPServer((args.host, args.port), MemoryHTTPHandler)
    print(
        json_dumps(
            {
                "ok": True,
                "db": str(args.db),
                "url": f"http://{args.host}:{args.port}",
                "model": MemoryHTTPHandler.model_config.model,
                "endpoint": MemoryHTTPHandler.model_config.endpoint,
                "no_llm": MemoryHTTPHandler.model_config.no_llm,
            },
            indent=2,
        )
    )
    server.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "init-db":
        return command_init_db(args)
    if args.command == "ingest":
        return command_ingest(args)
    if args.command == "compose":
        return command_compose(args)
    if args.command == "query":
        return command_query(args)
    if args.command == "merge-session":
        return command_merge_session(args)
    if args.command == "serve":
        return command_serve(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
