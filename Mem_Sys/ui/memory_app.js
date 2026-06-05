const serviceStatus = document.querySelector("#serviceStatus");
const modelPill = document.querySelector("#modelPill");
const dbPill = document.querySelector("#dbPill");
const formTab = document.querySelector("#formTab");
const jsonTab = document.querySelector("#jsonTab");
const memoryForm = document.querySelector("#memoryForm");
const jsonPane = document.querySelector("#jsonPane");
const jsonInput = document.querySelector("#jsonInput");
const extractButton = document.querySelector("#extractButton");
const extractJsonButton = document.querySelector("#extractJsonButton");
const syncFromForm = document.querySelector("#syncFromForm");
const resetButton = document.querySelector("#resetButton");
const samplePreference = document.querySelector("#samplePreference");
const sampleBlocked = document.querySelector("#sampleBlocked");
const refreshTiers = document.querySelector("#refreshTiers");
const mergeProfile = document.querySelector("#mergeProfile");
const refreshPeople = document.querySelector("#refreshPeople");
const personList = document.querySelector("#personList");
const gateStrip = document.querySelector("#gateStrip");
const atomsList = document.querySelector("#atomsList");
const rawLedgerList = document.querySelector("#rawLedgerList");
const shortTermList = document.querySelector("#shortTermList");
const profileList = document.querySelector("#profileList");
const rawOutput = document.querySelector("#rawOutput");
const lastEvent = document.querySelector("#lastEvent");
const tierStats = document.querySelector("#tierStats");
const dossierTitle = document.querySelector("#dossierTitle");

const fields = {
  personId: document.querySelector("#personId"),
  displayName: document.querySelector("#displayName"),
  speakerState: document.querySelector("#speakerState"),
  asrConfidence: document.querySelector("#asrConfidence"),
  utteranceText: document.querySelector("#utteranceText"),
  sceneSummary: document.querySelector("#sceneSummary"),
  replyText: document.querySelector("#replyText"),
  commandsJson: document.querySelector("#commandsJson"),
  allowMemory: document.querySelector("#allowMemory"),
  allowProfile: document.querySelector("#allowProfile"),
  needClarification: document.querySelector("#needClarification"),
};

let config = {
  model: "qwen3.5-2b-q4_k_m",
  endpoint: "http://127.0.0.1:8081/v1/chat/completions",
  no_llm: false,
  db: "",
};
let currentPersonId = "person_demo_liuzhe";
let currentSessionId = "";
let peopleCache = [];

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function stablePart(value) {
  return String(value || "")
    .trim()
    .replace(/[^\w.-]+/g, "_")
    .slice(0, 36) || "demo";
}

function parseCommands() {
  const raw = fields.commandsJson.value.trim();
  if (!raw) return [];
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error("Commands JSON must be an array");
  }
  return parsed;
}

function buildEnvelope() {
  const stamp = Date.now();
  const personId = fields.personId.value.trim() || "person_demo";
  const turnId = `ui_turn_${stamp}`;
  const text = fields.utteranceText.value.trim();
  const commands = parseCommands();
  return {
    schema: "robot.memory.input.v1",
    event_id: `mem_in_ui_${stablePart(personId)}_${stamp}`,
    session_id: `ui_session_${new Date().toISOString().slice(0, 10)}`,
    turn_id: turnId,
    household_id: "local_household",
    robot_id: "local_robot",
    ts_ms: stamp,
    subject: {
      person_id: personId,
      display_name: fields.displayName.value.trim(),
      speaker_state: fields.speakerState.value,
      identity_context: {
        person_id: personId,
        display_name: fields.displayName.value.trim(),
        voiceprint_id: "ui_voiceprint_demo",
        voiceprint_state: "matched",
        spk_state: "spk_voiceprint_linked",
        source: "ui_demo",
      },
      speaker_source: {
        state: fields.speakerState.value,
        model: "ui_demo",
        selection_reason: "manual_demo",
      },
      speaker_probabilities: [],
    },
    utterance: {
      text,
      language: "zh",
      start_ms: stamp - 2600,
      end_ms: stamp,
      asr_confidence: Number(fields.asrConfidence.value || 0),
      segments: [
        {
          text,
          turn_id: turnId,
          speaker_confidence: Number(fields.asrConfidence.value || 0),
          voiceprint_state: "matched",
        },
      ],
    },
    fast_reaction: {
      speaker_turn: {
        schema: "face_lab.speaker_turn.v1",
        type: "speaker_turn.revised",
        session_id: `ui_session_${new Date().toISOString().slice(0, 10)}`,
        turn_id: turnId,
        person_id: personId,
        display_name: fields.displayName.value.trim(),
        speaker_state: fields.speakerState.value,
        confidence: Number(fields.asrConfidence.value || 0),
        voiceprint_id: "ui_voiceprint_demo",
        voiceprint_state: "matched",
        spk_state: "spk_voiceprint_linked",
        evidence: {
          conflict: false,
          overlap: fields.speakerState.value === "overlap_speech",
          mixed_speech_risk: fields.speakerState.value === "overlap_speech",
        },
        policy: {
          allow_memory_write: fields.allowMemory.checked,
          need_clarification: fields.needClarification.checked,
        },
        allow_profile_write: fields.allowProfile.checked,
        profile_write_reason: fields.allowProfile.checked ? "" : "ui_profile_write_disabled",
      },
      audio_event: {
        schema: "face_lab.audio.v1",
        type: "vad_off",
        turn_id: turnId,
        asr_text: text,
        language: "zh",
        voiceprint_id: "ui_voiceprint_demo",
        voiceprint_state: "matched",
        mixed_speech_risk: fields.speakerState.value === "overlap_speech",
      },
      visual: {},
      nlu: {},
      policy: {
        allow_memory_write: fields.allowMemory.checked,
        allow_profile_write: fields.allowProfile.checked,
        need_clarification: fields.needClarification.checked,
        profile_write_reason: fields.allowProfile.checked ? "" : "ui_profile_write_disabled",
      },
    },
    slow_reaction: {
      scene: {
        summary: fields.sceneSummary.value.trim(),
        activity: "",
        objects: [],
        room: "",
        people: [],
      },
      environment: {},
      scene_model: config.model,
    },
    robot_interaction: {
      reply_text: fields.replyText.value.trim(),
      commands,
      scheduler_result: {
        schema: "face_lab.interaction.scheduler.v1",
        type: "interaction_scheduler.result",
        ok: true,
        command_results: commands,
      },
      proactive_candidate: {},
      execution_result: {},
    },
    source_refs: {
      raw_event_ids: ["ui_demo.voice_turn", "ui_demo.slow_scene"],
      snapshot_refs: [],
      audio_refs: [],
    },
  };
}

function setMode(mode) {
  const formMode = mode === "form";
  memoryForm.classList.toggle("hidden", !formMode);
  jsonPane.classList.toggle("hidden", formMode);
  formTab.classList.toggle("active", formMode);
  jsonTab.classList.toggle("active", !formMode);
  if (!formMode) {
    jsonInput.value = formatJson(buildEnvelope());
  }
}

function setBusy(busy) {
  extractButton.disabled = busy;
  extractJsonButton.disabled = busy;
  mergeProfile.disabled = busy;
  refreshTiers.disabled = busy;
  refreshPeople.disabled = busy;
  serviceStatus.textContent = busy ? "extracting" : "ready";
}

function setGate(result) {
  const allowed = Boolean(result?.write_gate?.allowed);
  const reasons = result?.write_gate?.reasons || [];
  gateStrip.classList.toggle("allowed", allowed);
  gateStrip.classList.toggle("blocked", !allowed);
  gateStrip.querySelector("strong").textContent = allowed
    ? "allowed"
    : reasons.length
      ? reasons.join(", ")
      : "blocked";
}

function renderMemoryList(container, items, renderItem) {
  container.innerHTML = "";
  if (!items || !items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "empty";
    container.appendChild(empty);
    return;
  }
  for (const item of items) {
    container.appendChild(renderItem(item));
  }
}

function memoryCard(title, body, meta = "") {
  const item = document.createElement("article");
  item.className = "memory-card";
  const top = document.createElement("div");
  top.className = "memory-card-top";
  const heading = document.createElement("strong");
  heading.textContent = title;
  const metaNode = document.createElement("span");
  metaNode.textContent = meta;
  const text = document.createElement("p");
  text.textContent = body || "(empty)";
  top.append(heading, metaNode);
  item.append(top, text);
  return item;
}

function renderRawLedger(items) {
  renderMemoryList(rawLedgerList, items, (event) => {
    const gate = event.allow_memory_write || event.allow_profile_write ? "gate:on" : "gate:off";
    const meta = `${event.speaker_state} · ${gate}`;
    const body = [
      event.utterance_text ? `用户: ${event.utterance_text}` : "",
      event.scene_summary ? `场景: ${event.scene_summary}` : "",
      event.robot_reply ? `机器人: ${event.robot_reply}` : "",
    ].filter(Boolean).join(" | ");
    return memoryCard(event.turn_id || event.event_id, body, meta);
  });
}

function renderShortTerm(items) {
  renderMemoryList(shortTermList, items, (item) =>
    memoryCard(item.state_type, item.summary, item.turn_id),
  );
}

function renderPeople(people) {
  personList.innerHTML = "";
  if (!people || !people.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "还没有人物档案";
    personList.appendChild(empty);
    return;
  }
  for (const person of people) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "person-item";
    button.classList.toggle("active", person.person_id === currentPersonId);

    const name = document.createElement("strong");
    name.textContent = person.display_name || person.person_id;
    const id = document.createElement("span");
    id.textContent = person.person_id;
    const counts = document.createElement("small");
    const c = person.counts || {};
    counts.textContent = `短 ${c.short_term || 0} · 中 ${c.mid_term || 0} · 长 ${c.long_term || 0}`;

    button.append(name, id, counts);
    button.addEventListener("click", () => selectPerson(person));
    personList.appendChild(button);
  }
}

function selectPerson(person) {
  currentPersonId = person.person_id;
  currentSessionId = "";
  fields.personId.value = person.person_id;
  fields.displayName.value = person.display_name || "";
  renderPeople(peopleCache);
  loadTiers(currentPersonId, "");
}

function renderAtoms(atoms) {
  atomsList.innerHTML = "";
  if (!atoms || !atoms.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "no inserted atoms";
    atomsList.appendChild(empty);
    return;
  }
  for (const atom of atoms) {
    const item = document.createElement("article");
    item.className = "atom";
    const top = document.createElement("div");
    top.className = "atom-top";
    const type = document.createElement("span");
    type.className = "atom-type";
    type.textContent = atom.type;
    const score = document.createElement("span");
    score.className = "atom-score";
    score.textContent = `${atom.subject_scope}/${atom.subject_id} · c${Number(atom.confidence).toFixed(2)} · i${Number(atom.importance).toFixed(2)}`;
    const text = document.createElement("p");
    text.textContent = atom.text;
    top.append(type, score);
    item.append(top, text);
    atomsList.appendChild(item);
  }
}

function renderTierStats(stats) {
  tierStats.innerHTML = "";
  const values = [
    ["L0", stats?.l0_raw_events || 0],
    ["短期", stats?.short_term_items || 0],
    ["中期", stats?.mid_term_atoms || 0],
    ["长期", stats?.long_term_profile_items || 0],
    ["待确认", stats?.confirmation_candidates || 0],
  ];
  for (const [label, value] of values) {
    const item = document.createElement("span");
    item.textContent = `${label} ${value}`;
    tierStats.appendChild(item);
  }
}

function profileEntryText(entry) {
  if (typeof entry === "string") return entry;
  if (!entry || typeof entry !== "object") return String(entry || "");
  return entry.text || entry.name || entry.value || JSON.stringify(entry);
}

function renderProfile(profile) {
  profileList.innerHTML = "";
  if (!profile || !Object.keys(profile).length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "还没有长期画像";
    profileList.appendChild(empty);
    return;
  }
  const sections = [
    ["aliases", "称呼"],
    ["stable_facts", "稳定事实"],
    ["preferences", "偏好"],
    ["communication_style", "沟通风格"],
    ["recent_context", "近期上下文"],
    ["boundaries", "边界/规则"],
    ["uncertain", "待确认"],
  ];
  let rendered = 0;
  for (const [key, label] of sections) {
    const values = Array.isArray(profile[key]) ? profile[key] : [];
    for (const value of values) {
      const card = memoryCard(label, profileEntryText(value), value.memory_id || "");
      profileList.appendChild(card);
      rendered++;
    }
  }
  if (!rendered) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "长期画像为空";
    profileList.appendChild(empty);
  }
}

async function loadPeople() {
  try {
    const response = await fetch("/memory/people");
    const body = await response.json();
    if (!response.ok || !body.ok) throw new Error(body.error || response.statusText);
    peopleCache = body.people || [];
    if (!peopleCache.some((person) => person.person_id === currentPersonId) && peopleCache.length) {
      currentPersonId = peopleCache[0].person_id;
      fields.personId.value = peopleCache[0].person_id;
      fields.displayName.value = peopleCache[0].display_name || "";
    }
    renderPeople(peopleCache);
  } catch (error) {
    personList.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = `人物列表加载失败: ${error.message}`;
    personList.appendChild(empty);
  }
}

async function fetchConfig() {
  try {
    const response = await fetch("/memory/config");
    const body = await response.json();
    if (!response.ok || !body.ok) throw new Error(body.error || response.statusText);
    config = body;
    serviceStatus.textContent = body.no_llm ? "ready · rules" : "ready · model";
    modelPill.textContent = body.no_llm ? `${body.model} · rules` : body.model;
    dbPill.textContent = body.db || "memory.sqlite3";
  } catch (error) {
    serviceStatus.textContent = `offline · ${error.message}`;
    modelPill.textContent = "qwen3.5-2b-q4_k_m";
    dbPill.textContent = "not connected";
  }
}

async function postEnvelope(envelope) {
  setBusy(true);
  rawOutput.textContent = formatJson({ envelope });
  currentPersonId = envelope.subject?.person_id || currentPersonId;
  currentSessionId = envelope.session_id || currentSessionId;
  try {
    const response = await fetch("/memory/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(envelope),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || response.statusText);
    }
    setGate(result);
    lastEvent.textContent = `${result.event_id} · ${result.extractor}`;
    rawOutput.textContent = formatJson({ result, envelope });
    await loadPeople();
    await loadTiers(currentPersonId, "");
  } catch (error) {
    gateStrip.classList.remove("allowed");
    gateStrip.classList.add("blocked");
    gateStrip.querySelector("strong").textContent = error.message;
    rawOutput.textContent = formatJson({ error: error.message, envelope });
  } finally {
    setBusy(false);
  }
}

async function loadTiers(personId = currentPersonId, sessionId = currentSessionId) {
  if (!personId) return;
  try {
    const response = await fetch("/memory/tier-dashboard", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ person_id: personId, session_id: sessionId, limit: 30 }),
    });
    const body = await response.json();
    if (!response.ok || !body.ok) throw new Error(body.error || response.statusText);
    const tiers = body.tiers || {};
    renderTierStats(body.stats || {});
    renderRawLedger(tiers.l0_raw_ledger || []);
    renderShortTerm(tiers.short_term || []);
    renderAtoms(tiers.mid_term_atoms || []);
    renderProfile(tiers.long_term_profile || {});
    const matched = peopleCache.find((person) => person.person_id === personId);
    dossierTitle.textContent = matched
      ? `${matched.display_name || personId} 的记忆档案`
      : `${personId} 的记忆档案`;
  } catch (error) {
    renderProfile({ uncertain: [`档案加载失败: ${error.message}`] });
  }
}

async function mergeLongTermProfile() {
  const personId = fields.personId.value.trim() || currentPersonId;
  const sessionId = "";
  if (!personId) return;
  setBusy(true);
  try {
    const response = await fetch("/memory/profile-merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ person_id: personId, session_id: sessionId }),
    });
    const body = await response.json();
    if (!response.ok || !body.ok) throw new Error(body.error || response.statusText);
    renderProfile(body.profile || {});
    rawOutput.textContent = formatJson({ profile_merge: body });
    await loadPeople();
    await loadTiers(personId, "");
  } catch (error) {
    rawOutput.textContent = formatJson({ error: error.message });
  } finally {
    setBusy(false);
  }
}

function resetForm() {
  fields.personId.value = "person_demo_liuzhe";
  fields.displayName.value = "刘哲";
  fields.speakerState.value = "known_visible_speaker";
  fields.asrConfidence.value = "0.92";
  fields.utteranceText.value = "明天下午三点提醒我开产品评审会，我喜欢会议前十分钟收到提醒。";
  fields.sceneSummary.value = "用户坐在办公桌前，面前有笔记本电脑和两部手机，正在处理工作。";
  fields.replyText.value = "好的，我会在明天下午三点前提醒你。";
  fields.commandsJson.value = formatJson([
    { command: "schedule_reminder", ok: true, message: "created reminder for product review" },
  ]);
  fields.allowMemory.checked = true;
  fields.allowProfile.checked = true;
  fields.needClarification.checked = false;
  jsonInput.value = formatJson(buildEnvelope());
}

function fillPreferenceSample() {
  fields.speakerState.value = "known_visible_speaker";
  fields.allowMemory.checked = true;
  fields.allowProfile.checked = true;
  fields.needClarification.checked = false;
  fields.utteranceText.value = "我喜欢喝冰美式，不喜欢太甜的咖啡，以后帮我点咖啡时少糖。";
  fields.sceneSummary.value = "用户站在厨房岛台旁边，手里拿着咖啡杯。";
  fields.replyText.value = "记住了，咖啡偏好是冰美式和少糖。";
  fields.commandsJson.value = "[]";
  jsonInput.value = formatJson(buildEnvelope());
}

function fillBlockedSample() {
  fields.speakerState.value = "uncertain";
  fields.allowMemory.checked = false;
  fields.allowProfile.checked = false;
  fields.needClarification.checked = true;
  fields.utteranceText.value = "明天下午提醒我去取快递。";
  fields.sceneSummary.value = "画面中有两个人，当前说话人身份未确认。";
  fields.replyText.value = "";
  fields.commandsJson.value = "[]";
  jsonInput.value = formatJson(buildEnvelope());
}

memoryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await postEnvelope(buildEnvelope());
  } catch (error) {
    rawOutput.textContent = formatJson({ error: error.message });
  }
});

extractJsonButton.addEventListener("click", async () => {
  try {
    await postEnvelope(JSON.parse(jsonInput.value));
  } catch (error) {
    rawOutput.textContent = formatJson({ error: error.message });
  }
});

syncFromForm.addEventListener("click", () => {
  jsonInput.value = formatJson(buildEnvelope());
});

formTab.addEventListener("click", () => setMode("form"));
jsonTab.addEventListener("click", () => setMode("json"));
resetButton.addEventListener("click", resetForm);
samplePreference.addEventListener("click", fillPreferenceSample);
sampleBlocked.addEventListener("click", fillBlockedSample);
refreshTiers.addEventListener("click", () => {
  currentPersonId = fields.personId.value.trim() || currentPersonId;
  loadTiers(currentPersonId, "");
});
mergeProfile.addEventListener("click", mergeLongTermProfile);
refreshPeople.addEventListener("click", loadPeople);

resetForm();
fetchConfig();
loadPeople().then(() => loadTiers(currentPersonId, ""));
