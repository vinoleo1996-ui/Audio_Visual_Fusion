#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import pathlib
import sqlite3
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "memory_service.py"
SPEC = importlib.util.spec_from_file_location("memory_service", MODULE_PATH)
memory_service = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["memory_service"] = memory_service
SPEC.loader.exec_module(memory_service)


def envelope(
    *,
    event_id: str = "mem_in_test_1",
    speaker_state: str = "known_visible_speaker",
    allow_memory_write: bool = True,
    allow_profile_write: bool = True,
    need_clarification: bool = False,
    text: str = "明天下午三点提醒我开会",
    slow_summary: str = "",
) -> dict:
    return {
        "schema": "robot.memory.input.v1",
        "event_id": event_id,
        "session_id": "voice_session_test",
        "turn_id": "audio_turn_test",
        "household_id": "local_household",
        "robot_id": "local_robot",
        "ts_ms": 1779783365639,
        "subject": {
            "person_id": "person_a",
            "display_name": "刘哲",
            "speaker_state": speaker_state,
            "identity_context": {
                "voiceprint_id": "voiceprint_a",
                "voiceprint_state": "matched",
                "spk_state": "spk_voiceprint_linked",
            },
            "speaker_source": {},
            "speaker_probabilities": [],
        },
        "utterance": {
            "text": text,
            "language": "zh",
            "start_ms": 1779783363000,
            "end_ms": 1779783365639,
            "asr_confidence": 0.9,
            "segments": [],
        },
        "fast_reaction": {
            "speaker_turn": {
                "evidence": {"conflict": False, "overlap": False},
                "voiceprint_id": "voiceprint_a",
                "voiceprint_state": "matched",
                "spk_state": "spk_voiceprint_linked",
            },
            "audio_event": {"asr_text": text, "voiceprint_state": "matched"},
            "visual": {},
            "nlu": {},
            "policy": {
                "allow_memory_write": allow_memory_write,
                "allow_profile_write": allow_profile_write,
                "need_clarification": need_clarification,
                "profile_write_reason": "",
            },
        },
        "slow_reaction": {
            "scene": {
                "summary": slow_summary,
                "activity": "",
                "objects": [],
                "room": "",
                "people": [],
            },
            "environment": {},
            "scene_model": "qwen3.5-4b",
        },
        "robot_interaction": {
            "reply_text": "",
            "commands": [],
            "scheduler_result": {},
            "proactive_candidate": {},
            "execution_result": {},
        },
        "source_refs": {"raw_event_ids": [], "snapshot_refs": [], "audio_refs": []},
    }


class MemoryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = pathlib.Path(self.tmp.name) / "memory.sqlite3"
        self.conn = memory_service.connect_db(self.db_path)
        memory_service.init_db(self.conn)
        self.config = memory_service.ModelConfig(no_llm=True)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_disallowed_uncertain_turn_writes_only_l0(self) -> None:
        result = memory_service.ingest_envelope(
            self.conn,
            envelope(
                speaker_state="uncertain",
                allow_memory_write=False,
                allow_profile_write=False,
                need_clarification=True,
                text="古0嘛需要吗？对对。",
            ),
            self.config,
        )

        self.assertFalse(result["personal_write_allowed"])
        self.assertEqual(self.count("raw_memory_events"), 1)
        self.assertGreaterEqual(self.count("short_term_state"), 1)
        self.assertEqual(self.count("memory_atoms"), 0)
        self.assertEqual(self.count("person_profiles"), 0)
        self.assertEqual(self.count("confirmation_candidates"), 1)

    def test_allowed_known_visible_generates_atom_without_llm(self) -> None:
        result = memory_service.ingest_envelope(self.conn, envelope(), self.config)

        self.assertTrue(result["personal_write_allowed"])
        self.assertEqual(self.count("raw_memory_events"), 1)
        self.assertGreaterEqual(self.count("memory_atoms"), 1)
        self.assertEqual(self.count("person_profiles"), 1)
        row = self.conn.execute("SELECT type, text FROM memory_atoms LIMIT 1").fetchone()
        self.assertIn(row["type"], {"commitment", "plan"})
        self.assertIn("提醒", row["text"])

    def test_slow_reaction_update_is_idempotent(self) -> None:
        first = envelope(event_id="mem_in_same_turn", slow_summary="")
        second = envelope(event_id="mem_in_same_turn", slow_summary="用户坐在办公桌前。")

        result1 = memory_service.ingest_envelope(self.conn, first, self.config)
        result2 = memory_service.ingest_envelope(self.conn, second, self.config)

        self.assertFalse(result1["raw_event_existed"])
        self.assertTrue(result2["raw_event_existed"])
        self.assertEqual(self.count("raw_memory_events"), 1)
        self.assertEqual(result2["inserted_atom_count"], 0)
        row = self.conn.execute("SELECT raw_json FROM raw_memory_events").fetchone()
        self.assertIn("办公桌", row["raw_json"])

    def test_overlap_does_not_pollute_profile(self) -> None:
        result = memory_service.ingest_envelope(
            self.conn,
            envelope(speaker_state="overlap_speech", text="我喜欢喝拿铁"),
            self.config,
        )

        self.assertFalse(result["personal_write_allowed"])
        self.assertIn("overlap_speech", result["write_gate"]["reasons"])
        self.assertEqual(self.count("memory_atoms"), 0)
        self.assertEqual(self.count("person_profiles"), 0)

    def test_sensitive_requires_explicit_memory_request(self) -> None:
        result = memory_service.ingest_envelope(
            self.conn,
            envelope(text="我的身份证号码是123456"),
            self.config,
        )

        self.assertFalse(result["personal_write_allowed"])
        self.assertIn(
            "sensitive_without_explicit_memory_request",
            result["write_gate"]["reasons"],
        )
        self.assertEqual(self.count("memory_atoms"), 0)

    def test_compose_from_proactive_voice_state_respects_policy(self) -> None:
        voice_state = {
            "last_turn": {
                "schema": "face_lab.voice.interaction.v1",
                "type": "voice.turn",
                "session_id": "voice_session_x",
                "turn_id": "audio_turn_1",
                "person_id": "person_a",
                "display_name": "刘哲",
                "text": "我喜欢喝茶",
                "audio_event": {
                    "ts_ms": 1779783365639,
                    "language": "zh",
                    "start_ms": 1,
                    "end_ms": 2,
                    "asr_text": "我喜欢喝茶",
                },
                "speaker_turn": {
                    "type": "speaker_turn.revised",
                    "speaker_state": "uncertain",
                    "allow_profile_write": False,
                    "profile_write_reason": "profile_write_requires_identity_commit",
                    "policy": {
                        "allow_memory_write": False,
                        "need_clarification": True,
                    },
                    "identity_context": {},
                    "speaker_source": {"state": "uncertain"},
                    "speaker_probabilities": [],
                },
            }
        }
        composed = memory_service.compose_from_proactive_os(voice_state=voice_state)
        result = memory_service.ingest_envelope(self.conn, composed, self.config)

        self.assertEqual(composed["schema"], "robot.memory.input.v1")
        self.assertEqual(composed["subject"]["speaker_state"], "uncertain")
        self.assertFalse(composed["fast_reaction"]["policy"]["allow_memory_write"])
        self.assertFalse(result["personal_write_allowed"])
        self.assertEqual(self.count("memory_atoms"), 0)

    def test_tier_dashboard_returns_short_mid_long_layers(self) -> None:
        memory_service.ingest_envelope(self.conn, envelope(), self.config)

        dashboard = memory_service.query_memory_tiers(
            self.conn,
            person_id="person_a",
            session_id="voice_session_test",
            limit=20,
        )

        self.assertTrue(dashboard["ok"])
        self.assertEqual(dashboard["stats"]["l0_raw_events"], 1)
        self.assertGreaterEqual(dashboard["stats"]["short_term_items"], 1)
        self.assertGreaterEqual(dashboard["stats"]["mid_term_atoms"], 1)
        self.assertGreaterEqual(dashboard["stats"]["long_term_profile_items"], 1)
        self.assertIn("l0_raw_ledger", dashboard["tiers"])
        self.assertIn("short_term", dashboard["tiers"])
        self.assertIn("mid_term_atoms", dashboard["tiers"])
        self.assertIn("long_term_profile", dashboard["tiers"])

    def test_people_list_is_person_centered(self) -> None:
        memory_service.ingest_envelope(self.conn, envelope(), self.config)

        people = memory_service.list_people(self.conn)

        self.assertTrue(people["ok"])
        self.assertEqual(len(people["people"]), 1)
        person = people["people"][0]
        self.assertEqual(person["person_id"], "person_a")
        self.assertEqual(person["display_name"], "刘哲")
        self.assertGreaterEqual(person["counts"]["short_term"], 1)
        self.assertGreaterEqual(person["counts"]["mid_term"], 1)
        self.assertGreaterEqual(person["counts"]["long_term"], 1)


if __name__ == "__main__":
    unittest.main()
