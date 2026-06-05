# Mem_Sys: 端侧人物记忆系统

`Mem_Sys` 是家庭机器人端侧记忆系统的独立模块。它消费上游快反应/慢反应聚合后的标准 JSON envelope，不重新猜身份，并按人物形成独立的短期、中期、长期记忆档案。

## 目录

```text
Mem_Sys/
├── README.md
├── docs/
│   └── memory_service.md
├── scripts/
│   ├── memory_service.py
│   └── start_memory_lab.sh
├── tests/
│   └── test_memory_service.py
└── ui/
    ├── memory.html
    ├── memory_app.js
    └── memory_styles.css
```

## 核心能力

- 标准输入：`robot.memory.input.v1`
- L0 原始账本：完整保存 envelope 和上游门控证据
- 短期记忆：规则生成，保存最近 turn、任务状态、机器人命令、待确认事项
- 中期记忆：Qwen3.5-2B 抽取 `memory_atom`
- 长期记忆：按 `person_id` 聚合为人物 profile，可手动触发 session/person 级归纳
- 人物档案：每个人独立展示短期、中期、长期记忆
- 写入门控：严格遵守 `allow_memory_write / allow_profile_write / need_clarification / speaker_state`
- 本地运行：SQLite + llama.cpp `llama-server`，不依赖云服务

## 一行启动

```bash
cd /home/nvidia/Documents/ASD_Fusion/Mem_Sys && bash scripts/start_memory_lab.sh --open
```

脚本会自动：

- 复用或启动 Qwen3.5-2B `llama-server`
- 自动寻找不冲突的模型端口和 UI 端口
- 启动 Memory UI 后端
- 输出本机 URL 和局域网 URL

默认模型路径：

```text
/home/nvidia/Documents/Models/Qwen3.5-2B-Q4_K_M/Qwen3.5-2B-Q4_K_M.gguf
```

默认模型名：

```text
qwen3.5-2b-q4_k_m
```

## 手动启动

启动模型服务：

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
```

启动记忆服务：

```bash
cd /home/nvidia/Documents/ASD_Fusion/Mem_Sys
python3 scripts/memory_service.py \
  --db output/memory/memory.sqlite3 \
  serve \
  --host 0.0.0.0 \
  --port 8095 \
  --endpoint http://127.0.0.1:8081/v1/chat/completions \
  --model qwen3.5-2b-q4_k_m \
  --timeout-s 60 \
  --fallback-on-llm-error
```

打开：

```text
http://<设备 IP>:8095/memory/ui/
```

## API

- `GET /memory/ui/`: 人物记忆档案 UI
- `GET /memory/config`: 服务配置
- `GET /memory/people`: 人物档案列表
- `POST /memory/events`: 写入一个 `robot.memory.input.v1` envelope
- `POST /memory/tier-dashboard`: 查询某人的短期/中期/长期记忆档案
- `POST /memory/profile-merge`: 触发某人的长期画像归纳
- `POST /memory/context-pack`: 给对话/规划侧使用的记忆上下文检索
- `GET /memory/person/<person_id>/dashboard`: 简单人物 dashboard
- `GET /memory/person/<person_id>/tiers`: 人物分层档案

## 测试

```bash
cd /home/nvidia/Documents/ASD_Fusion/Mem_Sys
python3 -m unittest discover -s tests -p 'test_*.py'
node --check ui/memory_app.js
bash -n scripts/start_memory_lab.sh
```

## 写入门控

个人记忆写入必须满足：

- `allow_memory_write=true` 或 `allow_profile_write=true`
- `speaker_state=known_visible_speaker`，或 `known_offscreen_speaker` 且绑定声纹、无 AV conflict
- 非 `overlap_speech`
- `need_clarification=false`
- ASR 文本有效，不是明显噪声/误识别
- 敏感内容必须由用户明确要求记住

否则只写 L0 原始账本和短期待确认，不污染人物画像。

## 当前边界

当前版本已经实现端侧人物档案、短中长期分层、Qwen3.5-2B 抽取、SQLite 持久化和 UI 体验。后续仍需要补齐后台夜间归纳调度、冲突演化边 `supersedes/contradicts/supports` 的自动合并策略、跨 session 衰减和语义重排。
