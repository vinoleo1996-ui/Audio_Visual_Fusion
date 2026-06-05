# 端侧记忆服务

`scripts/memory_service.py` 是当前工程里的第一版端侧记忆萃取落地模块。它不重新判断“谁在说话”，只消费上游快慢反应聚合后的 `robot.memory.input.v1` envelope，并把原始事件、短期状态、中期 memory atom 和人物画像保存在本地 SQLite。

## 入口

标准输入 schema 是 `robot.memory.input.v1`。快系统提供身份、ASR、ASD、指令、执行状态和写入门控；慢系统补充场景、环境、物体、活动和上下文理解。记忆服务只读取 envelope 里的证据，并严格遵守：

- `fast_reaction.policy.allow_memory_write`
- `fast_reaction.policy.allow_profile_write`
- `fast_reaction.policy.need_clarification`
- `fast_reaction.policy.profile_write_reason`
- `subject.speaker_state`

同一个 `event_id/session_id/turn_id` 的慢反应后到时，可以再次调用 ingest。服务会 upsert L0 原始账本，并用确定性 `memory_id` 防止重复生成人物记忆。

## 命令

初始化数据库：

```bash
python3 scripts/memory_service.py --db output/memory/memory.sqlite3 init-db
```

从 ProactiveOS 当前输出组装 envelope：

```bash
python3 scripts/memory_service.py compose \
  --voice-state /home/nvidia/Documents/ProactiveOS_0.1/perception/face_lab/outputs/debug/voice_interaction_state.json \
  --slow-scene /home/nvidia/Documents/ProactiveOS_0.1/perception/face_lab/outputs/llm/scene_latest.json \
  --environment /home/nvidia/Documents/ProactiveOS_0.1/perception/face_lab/outputs/debug/environment_context_state.json \
  --scheduler-event /home/nvidia/Documents/ProactiveOS_0.1/perception/face_lab/outputs/debug/interaction_scheduler_events.ndjson \
  --proactive-state /home/nvidia/Documents/ProactiveOS_0.1/perception/face_lab/outputs/debug/proactive_interaction_state.json \
  --output output/memory/latest_envelope.json
```

写入 envelope。当前机器如果还没有 Qwen3.5 4B 或 llama-server 未启动，可先用规则 fallback 验证链路：

```bash
python3 scripts/memory_service.py --db output/memory/memory.sqlite3 ingest \
  --input output/memory/latest_envelope.json \
  --no-llm
```

接本地 Qwen3.5-2B llama-server：

```bash
python3 scripts/memory_service.py --db output/memory/memory.sqlite3 ingest \
  --input output/memory/latest_envelope.json \
  --endpoint http://127.0.0.1:8081/v1/chat/completions \
  --model qwen3.5-2b-q4_k_m
```

查询人物上下文：

```bash
python3 scripts/memory_service.py --db output/memory/memory.sqlite3 query \
  --person-id person_1779439367057 \
  --limit 20
```

启动 HTTP 服务：

```bash
python3 scripts/memory_service.py --db output/memory/memory.sqlite3 serve \
  --host 127.0.0.1 \
  --port 8095 \
  --endpoint http://127.0.0.1:8081/v1/chat/completions \
  --model qwen3.5-2b-q4_k_m
```

HTTP endpoints:

- `GET /memory/ui`: 端侧记忆萃取体验台
- `POST /memory/events`: body 为完整 `robot.memory.input.v1`
- `POST /memory/context-pack`: body 为 `{"person_id":"...", "text":"...", "limit":20}`
- `GET /memory/person/<person_id>/dashboard`

## 存储分层

- L0 raw ledger: `raw_memory_events` 原样保存 envelope。任何禁止写入、身份不确定、重叠说话、AV conflict、敏感但未明确要求记住的内容都至少落 L0。
- 短期记忆: `short_term_state` 由规则生成，包含最近 turn、机器人命令、NLU 槽位和待确认问题，不依赖 LLM。
- 中期记忆: `memory_atoms` 保存 `fact/preference/temporary_state/plan/commitment/alias/rule/action_experience`。
- 长期画像: `person_profiles` 按 atom 增量更新；`merge-session` 可按 session 触发 Qwen profile merge。
- 演化边: `memory_edges` 已建表，给后续 `supersedes/contradicts/supports` 合并策略使用。

## 写入门控

个人记忆写入必须同时满足：

- `allow_memory_write=true` 或 `allow_profile_write=true`
- `speaker_state=known_visible_speaker`，或 `known_offscreen_speaker` 且有绑定声纹并无 AV conflict
- 非 `overlap_speech`
- `need_clarification=false`
- ASR 文本有效且不是明显噪声
- 敏感内容必须包含明确记忆意图，例如“记住”或“帮我记一下”

不满足时服务仍写 L0 和短期状态，并尽量生成 `confirmation_candidate`，但不会更新 `memory_atoms/person_profiles`。

## 模型约束

默认模型参数：

- extractor: `temperature=0`, `max_tokens=768`
- profile merge: `temperature=0.1`, `max_tokens=1024`

调用协议使用 llama-server 的 OpenAI-compatible `/v1/chat/completions`。请求带 `response_format={"type":"json_object"}`；如果当前 llama.cpp 服务不支持强 schema/grammar，脚本会做 JSON parse 和一次 repair retry。LLM 失败会写 `model_call_logs`，只有显式传 `--fallback-on-llm-error` 才会自动退回规则抽取。

当前默认使用 `/home/nvidia/Documents/Models/Qwen3.5-2B-Q4_K_M/Qwen3.5-2B-Q4_K_M.gguf`，模型名为 `qwen3.5-2b-q4_k_m`。如果后续补齐 4B GGUF，只需要改 `--model` 和 llama-server 启动参数。

本地体验台启动示例：

```bash
/home/nvidia/Documents/ProactiveOS_0.1/perception/face_lab/third_party/llama.cpp/build/bin/llama-server \
  -m /home/nvidia/Documents/Models/Qwen3.5-2B-Q4_K_M/Qwen3.5-2B-Q4_K_M.gguf \
  --alias qwen3.5-2b-q4_k_m \
  --host 127.0.0.1 \
  --port 8081 \
  --ctx-size 4096 \
  --threads 6 \
  --gpu-layers 999 \
  --reasoning off

python3 scripts/memory_service.py --db output/memory/memory.sqlite3 serve \
  --host 127.0.0.1 \
  --port 8095 \
  --endpoint http://127.0.0.1:8081/v1/chat/completions \
  --model qwen3.5-2b-q4_k_m \
  --timeout-s 60 \
  --fallback-on-llm-error
```
