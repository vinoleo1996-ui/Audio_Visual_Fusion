const backendVideo = document.querySelector("#backendVideo");
const overlay = document.querySelector("#overlay");
const videoCanvas = document.querySelector("#videoCanvas");
const videoCtx = videoCanvas ? videoCanvas.getContext("2d") : null;
const ctx = overlay.getContext("2d");
const chatList = document.querySelector("#chatList");
const runtimeStatus = document.querySelector("#runtimeStatus");
const speakerState = document.querySelector("#speakerState");
const latencyState = document.querySelector("#latencyState");
const modeState = document.querySelector("#modeState");
const eventState = document.querySelector("#eventState");
const confidenceState = document.querySelector("#confidenceState");
const clearChat = document.querySelector("#clearChat");
const syncState = document.querySelector("#syncState");
const liveSubtitle = document.querySelector("#liveSubtitle");

let tracks = [];
let eventSource = null;
let frameStart = performance.now();
let lastEventTimestamp = 0;
let backendStreamActive = false;
let backendRenderBbox = false;
let videoReloadCount = 0;
let lastVideoLoadAt = performance.now();
let lastVideoActivityAt = performance.now();
let lastVideoReloadAt = 0;
let sourceWidth = 1280;
let sourceHeight = 720;
let activeThreshold = 0.65;
let tracksDirty = true;
let activeBubbles = {};
let wsVideo = null;
let wsVideoConnected = false;
let wsReconnectTimer = null;
let wsFrameCount = 0;
let wsDecodeInFlight = false;
let wsPendingFrame = null;
let wsLastRenderedSequence = 0;
let wsDroppedFrames = 0;
let previewAgeMs = 0;
const maxPreviewFrameAgeMs = 200;
let latestPartialUtteranceId = null;

const qualityColor = {
  good: "#46d46a",
  ok: "#f0d34e",
  low: "#8e969e",
};

function getBackendUrl(path) {
  if (window.location.protocol === "file:") {
    return `http://127.0.0.1:7050${path}`;
  }
  return path;
}

async function startVideoPreview() {
  // Try WebSocket+Canvas first (reliable), fall back to MJPEG <img>
  if (videoCanvas && videoCtx) {
    connectWebSocketVideo();
  } else {
    startMjpegFallback();
  }
  // Unified watchdog monitoring
  setInterval(checkVideoWatchdog, 3000);
}

let useCreateImageBitmap = true; // Use performant createImageBitmap with correct MIME-type blobs

function drawBlobWithObjectURL(blob, frameInfo) {
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    if (videoCanvas.width !== img.width || videoCanvas.height !== img.height) {
      videoCanvas.width = img.width;
      videoCanvas.height = img.height;
      sourceWidth = img.width;
      sourceHeight = img.height;
    }
    if (videoCtx) {
      renderPreviewImage(img, frameInfo);
    }
    URL.revokeObjectURL(url);
  };
  img.onerror = (imgErr) => {
    console.error("Image loading failed:", imgErr);
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function parsePreviewEnvelope(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 44 ||
      view.getUint8(0) !== 0x53 || view.getUint8(1) !== 0x50 ||
      view.getUint8(2) !== 0x4b || view.getUint8(3) !== 0x56 ||
      view.getUint8(4) !== 1) {
    return {
      sequence: ++wsFrameCount,
      captureEpochMs: Date.now(),
      jpeg: buffer,
    };
  }
  const sequence = Number(view.getBigUint64(8, true));
  const captureMonotonicMs = Number(view.getBigInt64(16, true));
  const sendMonotonicMs = Number(view.getBigInt64(24, true));
  const sendEpochMs = Number(view.getBigInt64(32, true));
  const jpegLength = view.getUint32(40, true);
  if (44 + jpegLength > view.byteLength) {
    return null;
  }
  return {
    sequence,
    captureEpochMs: sendEpochMs - (sendMonotonicMs - captureMonotonicMs),
    jpeg: buffer.slice(44, 44 + jpegLength),
  };
}

function renderPreviewImage(image, frameInfo) {
  if (!frameInfo || frameInfo.sequence <= wsLastRenderedSequence) {
    wsDroppedFrames++;
    return;
  }
  const ageMs = Math.max(0, Date.now() - frameInfo.captureEpochMs);
  if (ageMs > maxPreviewFrameAgeMs) {
    wsDroppedFrames++;
    return;
  }
  if (videoCanvas.width !== image.width || videoCanvas.height !== image.height) {
    videoCanvas.width = image.width;
    videoCanvas.height = image.height;
    sourceWidth = image.width;
    sourceHeight = image.height;
  }
  if (videoCtx) {
    videoCtx.drawImage(image, 0, 0);
  }
  wsLastRenderedSequence = frameInfo.sequence;
  previewAgeMs = ageMs;
  latencyState.textContent = `preview: ${Math.round(previewAgeMs)}ms · drop ${wsDroppedFrames}`;
}

function decodeLatestPreview() {
  if (wsDecodeInFlight || !wsPendingFrame) return;
  const frameInfo = wsPendingFrame;
  wsPendingFrame = null;
  if (frameInfo.sequence <= wsLastRenderedSequence ||
      Date.now() - frameInfo.captureEpochMs > maxPreviewFrameAgeMs) {
    wsDroppedFrames++;
    decodeLatestPreview();
    return;
  }
  wsDecodeInFlight = true;
  const blob = new Blob([frameInfo.jpeg], {type: "image/jpeg"});
  if (useCreateImageBitmap) {
    createImageBitmap(blob).then((bitmap) => {
      renderPreviewImage(bitmap, frameInfo);
      bitmap.close();
    }).catch((err) => {
      console.warn("createImageBitmap promise rejected, switching to URL.createObjectURL fallback:", err);
      useCreateImageBitmap = false;
      drawBlobWithObjectURL(blob, frameInfo);
    }).finally(() => {
      wsDecodeInFlight = false;
      decodeLatestPreview();
    });
  } else {
    drawBlobWithObjectURL(blob, frameInfo);
    wsDecodeInFlight = false;
    decodeLatestPreview();
  }
}

function connectWebSocketVideo() {
  if (wsVideo && wsVideo.readyState <= WebSocket.OPEN) return;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  let host = window.location.protocol === "file:" ? "127.0.0.1:7050" : window.location.host;
  if (window.location.port === "8080" || window.location.port === "7050") {
    host = `${window.location.hostname}:7200`;
  }
  const url = `${proto}//${host}/v1/video.ws`;
  runtimeStatus.textContent = "WebSocket video connecting...";

  wsVideo = new WebSocket(url);
  wsVideo.binaryType = "arraybuffer";

  wsVideo.onopen = () => {
    wsVideoConnected = true;
    backendStreamActive = true;
    lastVideoActivityAt = performance.now();
    runtimeStatus.textContent = "WebSocket video active";
    // Hide MJPEG fallback, show canvas
    if (backendVideo) backendVideo.style.display = "none";
    videoCanvas.style.display = "block";
  };

  wsVideo.onmessage = (event) => {
    lastVideoActivityAt = performance.now();
    wsFrameCount++;
    const frameInfo = parsePreviewEnvelope(event.data);
    if (!frameInfo) {
      wsDroppedFrames++;
      return;
    }
    if (wsPendingFrame) wsDroppedFrames++;
    wsPendingFrame = frameInfo;
    decodeLatestPreview();
  };

  wsVideo.onclose = () => {
    wsVideoConnected = false;
    backendStreamActive = false;
    runtimeStatus.textContent = "WebSocket video disconnected, falling back to MJPEG";
    startMjpegFallback();
  };

  wsVideo.onerror = () => {
    wsVideoConnected = false;
    runtimeStatus.textContent = "WebSocket video error, falling back to MJPEG";
    startMjpegFallback();
  };
}

function startMjpegFallback() {
  if (!backendVideo) return;
  // Show MJPEG img, hide canvas
  backendVideo.style.display = "block";
  if (videoCanvas) videoCanvas.style.display = "none";
  backendVideo.addEventListener("load", () => {
    backendStreamActive = true;
    lastVideoLoadAt = performance.now();
    lastVideoActivityAt = lastVideoLoadAt;
    runtimeStatus.textContent = "MJPEG detection stream active";
  });
  backendVideo.addEventListener("error", () => {
    backendStreamActive = false;
    runtimeStatus.textContent = "MJPEG stream unavailable";
    reloadVideoStream("error");
  });
  backendVideo.src = videoStreamUrl();
  runtimeStatus.textContent = "MJPEG detection stream connecting";
}

function videoStreamUrl() {
  return getBackendUrl(`/v1/video.mjpg?t=${Date.now()}&r=${videoReloadCount}`);
}

function reloadVideoStream(reason) {
  if (!backendVideo) return;
  const now = performance.now();
  if (now - lastVideoReloadAt < 2500) return;
  lastVideoReloadAt = now;
  videoReloadCount += 1;
  backendStreamActive = false;
  runtimeStatus.textContent = `Backend stream reconnecting (${reason})`;
  backendVideo.src = videoStreamUrl();
}

function checkVideoWatchdog() {
  const now = performance.now();
  const activityAge = lastVideoActivityAt ? now - lastVideoActivityAt : Infinity;
  const loadAge = lastVideoLoadAt ? now - lastVideoLoadAt : Infinity;
  
  if (wsVideoConnected) {
    // WebSocket mode active
    if (activityAge > 5000) {
      console.warn("WebSocket video feed stalled. Reconnecting...");
      runtimeStatus.textContent = "WebSocket stalled, reconnecting...";
      if (wsVideo) {
        try { wsVideo.close(); } catch(e) {}
      }
      connectWebSocketVideo();
    }
  } else if (backendVideo && backendVideo.style.display !== "none") {
    // MJPEG mode active
    const eventAge = lastEventTimestamp ? now - lastEventTimestamp : Infinity;
    if (!backendStreamActive && loadAge > 5000) {
      reloadVideoStream("startup");
      return;
    }
    if (backendStreamActive && eventAge > 12000 && activityAge > 12000) {
      reloadVideoStream("watchdog");
    }
  } else {
    // No stream active or startup phase stalled
    if (activityAge > 5000 && loadAge > 5000) {
      console.warn("No active video stream detected. Attempting recovery...");
      if (videoCanvas && videoCtx) {
        connectWebSocketVideo();
      } else {
        startMjpegFallback();
      }
    }
  }
}

function resizeOverlay() {
  const rect = overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.max(1, Math.round(rect.width * dpr));
  overlay.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawOverlay() {
  if (!tracksDirty) {
    requestAnimationFrame(drawOverlay);
    return;
  }
  tracksDirty = false;
  resizeOverlay();
  const rect = overlay.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  // Draw overlay boxes if backend stream is not active OR if backend does not render boxes directly
  if (!backendStreamActive || !backendRenderBbox) {
    for (const track of tracks) {
      const bbox = scaleBox(track.bbox, rect.width, rect.height);
      const isSpeaking = track.p_active > activeThreshold;
      
      // Determine quality color state from backend
      const render = track.render || {};
      const colorState = render.color_state || "low_quality";
      let color = qualityColor.low;
      if (colorState === "confirmed_good") {
        color = qualityColor.good;
      } else if (colorState === "tracking_ok") {
        color = qualityColor.ok;
      }
      
      const isLowQuality = colorState === "low_quality";
      const observationState = track.face?.observation_state ||
        (track.face?.face_bbox_observed === false ? "occluded" : "observed");
      const renderState = render.state || (observationState === "projected" ? "stable_tracking" : observationState);
      const isProjected = observationState === "projected";
      const isOccluded = observationState === "occluded" || observationState === "expired";
      
      // Dynamic speaking glow effect (only when quality is not low)
      if (isSpeaking && !isLowQuality) {
        ctx.shadowBlur = 10;
        ctx.shadowColor = "rgba(74, 163, 255, 0.45)";
      } else {
        ctx.shadowBlur = 0;
      }
      
      // Set line dash for occluded face tracks
      if (isProjected || isOccluded) {
        ctx.setLineDash([4, 4]);
      } else {
        ctx.setLineDash([]);
      }
      
      // Rounded bounding box
      ctx.lineWidth = isSpeaking ? 3 : 1.5;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.roundRect(bbox.x, bbox.y, bbox.w, bbox.h, 6);
      ctx.stroke();
      
      // Reset line dash
      ctx.setLineDash([]);
      
      // Draw 5-point landmarks if observed
      if (track.face && Array.isArray(track.face.landmarks_5pt) && observationState !== "expired") {
        ctx.fillStyle = color;
        for (const pt of track.face.landmarks_5pt) {
          const scaledPt = scalePoint(pt[0], pt[1], rect.width, rect.height);
          ctx.beginPath();
          ctx.arc(scaledPt.x, scaledPt.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      
      // Reset shadows for label drawing to avoid blurry text
      ctx.shadowBlur = 0;

      // Construct a modern status label
      const name = track.name || `P${track.person_track_id}`;
      let statusText = "";
      if (renderState === "stable_tracking" && isProjected) {
        statusText = "Tracking";
      } else if (renderState === "reacquiring") {
        statusText = "Reacquiring";
      } else if (renderState === "occluded" || isOccluded) {
        statusText = "Occluded";
      } else if (isSpeaking) {
        statusText = `Speaking ${Math.round(track.p_active * 100)}%`;
      } else {
        statusText = track.quality ? track.quality.toUpperCase() : "LOW";
      }
      const label = `${name} • ${statusText}`;

      ctx.font = "600 12px Inter, system-ui, -apple-system, sans-serif";
      const textMetrics = ctx.measureText(label);
      const paddingX = 8;
      const paddingY = 5;
      const textHeight = 12;
      
      const extraDotWidth = isSpeaking ? 12 : 0;
      const labelW = textMetrics.width + paddingX * 2 + extraDotWidth;
      const labelH = textHeight + paddingY * 2;
      
      // Position label above bbox
      const labelX = bbox.x;
      const labelY = Math.max(8, bbox.y - labelH - 5);

      // Card style background (glassmorphic look)
      ctx.fillStyle = "rgba(18, 22, 28, 0.82)";
      ctx.beginPath();
      ctx.roundRect(labelX, labelY, labelW, labelH, 4);
      ctx.fill();
      
      // Border around label card
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.stroke();

      // Text and speaking dot
      if (isSpeaking) {
        // Draw a pulse/speaking green/blue dot next to the label
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(labelX + paddingX + 4, labelY + paddingY + textHeight / 2, 3.5, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, labelX + paddingX + 12, labelY + paddingY + textHeight - 1);
      } else {
        ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
        ctx.fillText(label, labelX + paddingX, labelY + paddingY + textHeight - 1);
      }
    }
  }

  requestAnimationFrame(drawOverlay);
}

function scaleBox(bbox, width, height) {
  if (!bbox) {
    return { x: 0, y: 0, w: 0, h: 0 };
  }
  const videoRatio = sourceWidth / sourceHeight;
  const containerRatio = width / height;
  
  let scale, videoW, videoH, offsetX, offsetY;
  if (containerRatio > videoRatio) {
    // Pillarboxed (bars on left/right)
    scale = height / sourceHeight;
    videoW = sourceWidth * scale;
    videoH = height;
    offsetX = (width - videoW) / 2;
    offsetY = 0;
  } else {
    // Letterboxed (bars on top/bottom)
    scale = width / sourceWidth;
    videoW = width;
    videoH = sourceHeight * scale;
    offsetX = 0;
    offsetY = (height - videoH) / 2;
  }

  return {
    x: offsetX + (bbox.x1 / sourceWidth) * videoW,
    y: offsetY + (bbox.y1 / sourceHeight) * videoH,
    w: ((bbox.x2 - bbox.x1) / sourceWidth) * videoW,
    h: ((bbox.y2 - bbox.y1) / sourceHeight) * videoH,
  };
}

function scalePoint(pt_x, pt_y, width, height) {
  const videoRatio = sourceWidth / sourceHeight;
  const containerRatio = width / height;
  
  let scale, videoW, videoH, offsetX, offsetY;
  if (containerRatio > videoRatio) {
    scale = height / sourceHeight;
    videoW = sourceWidth * scale;
    videoH = height;
    offsetX = (width - videoW) / 2;
    offsetY = 0;
  } else {
    scale = width / sourceWidth;
    videoW = width;
    videoH = sourceHeight * scale;
    offsetX = 0;
    offsetY = (height - videoH) / 2;
  }

  return {
    x: offsetX + (pt_x / sourceWidth) * videoW,
    y: offsetY + (pt_y / sourceHeight) * videoH
  };
}

let reconnectTimeout = null;

function connectEvents() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }
  try {
    eventSource = new EventSource(getBackendUrl("/v1/fusion/events"));
    eventSource.onopen = () => {
      modeState.textContent = "mode: live";
      eventState.textContent = "live";
    };
    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        applyFusionEvent(payload);
      } catch (e) {
        console.warn("Malformed SSE event:", e, event.data);
      }
    };
    eventSource.onerror = () => {
      eventState.textContent = "reconnecting...";
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      reconnectTimeout = setTimeout(connectEvents, 2000);
    };
  } catch (_error) {
    eventState.textContent = "backend unavailable";
    reconnectTimeout = setTimeout(connectEvents, 5000);
  }
}

function applyFusionEvent(event) {
  frameStart = performance.now();
  lastEventTimestamp = performance.now();

  // Render system error/warnings dynamically
  if (event.fatal_error) {
    runtimeStatus.innerHTML = `<span style="color: #ff6e6e; font-weight: 600; text-shadow: 0 0 8px rgba(255,110,110,0.3);">Fatal: ${event.fatal_error}</span>`;
  } else if (event.latest_error) {
    runtimeStatus.innerHTML = `<span style="color: #f0d34e; font-weight: 600; text-shadow: 0 0 8px rgba(240,211,78,0.3);">Warning: ${event.latest_error}</span>`;
  } else if (backendStreamActive) {
    runtimeStatus.textContent = "Backend detection stream active";
  }

  if (event.type === "track_update" && Array.isArray(event.tracks)) {
    tracks = event.tracks.map((track) => {
      const current = tracks.find((item) => item.person_track_id === track.person_track_id);
      if (current &&
          (track.track_snapshot_sequence || 0) < (current.track_snapshot_sequence || 0)) {
        return current;
      }
      return track;
    });
    tracksDirty = true;
  }
  // Update source dimensions from hello event
  if (event.type === "hello") {
    if (event.width) sourceWidth = event.width;
    if (event.height) sourceHeight = event.height;
    if (event.asd_active_threshold !== undefined) activeThreshold = event.asd_active_threshold;
    if (event.render_bbox !== undefined) backendRenderBbox = event.render_bbox;
    return;
  }
  if (event.type === "heartbeat") {
    return;
  }
  if (event.type === "asd_update") {
    eventState.textContent = "asd";
    if (confidenceState && Array.isArray(event.tracks)) {
      const maxActive = event.tracks.reduce((value, track) => Math.max(value, track.p_active || 0), 0);
      confidenceState.textContent = `speaking: ${Math.round(maxActive * 100)}%`;
      const scores = new Map(event.tracks.map((track) => [track.person_track_id, track.p_active || 0]));
      tracks = tracks.map((track) => (
        scores.has(track.person_track_id)
          ? {...track, p_active: scores.get(track.person_track_id)}
          : track
      ));
      tracksDirty = true;
    }
    if (!event.text) {
      return;
    }
  }
  if (event.type === "asr_partial") {
    eventState.textContent = "partial asr";
    if (confidenceState) {
      confidenceState.textContent = `asr: ${Math.round((event.confidence || 0) * 100)}%`;
    }
    latestPartialUtteranceId = event.utterance_id;
    if (liveSubtitle) {
      liveSubtitle.textContent = event.text || "";
      liveSubtitle.classList.toggle("visible", Boolean(event.text));
    }
    addOrUpdateBubble(event);
    return;
  }
  if (event.final) {
    if (liveSubtitle) {
      liveSubtitle.textContent = "";
      liveSubtitle.classList.remove("visible");
    }
    if (latestPartialUtteranceId && activeBubbles[latestPartialUtteranceId] &&
        !activeBubbles[event.utterance_id]) {
      activeBubbles[event.utterance_id] = activeBubbles[latestPartialUtteranceId];
      delete activeBubbles[latestPartialUtteranceId];
    }
    latestPartialUtteranceId = null;
  }
  const ids = event.person_track_ids || [];
  const person = ids.length ? ids.map((id) => `P${id}`).join(", ") : event.position || "offscreen";
  speakerState.textContent = `speaker: ${person}`;
  if (confidenceState) {
    confidenceState.textContent = `confidence: ${Math.round((event.confidence || 0) * 100)}%`;
  }
  if (event.text) {
    addOrUpdateBubble(event);
  }
}

function getSpeakerTheme(speakerId) {
  const themes = {
    "P1": { bg: "rgba(74, 163, 255, 0.12)", border: "rgba(74, 163, 255, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #4aa3ff, #2f80ed)", avatarText: "#ffffff" },
    "P2": { bg: "rgba(240, 211, 78, 0.12)", border: "rgba(240, 211, 78, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #f0d34e, #d4af37)", avatarText: "#101316" },
    "P3": { bg: "rgba(149, 236, 105, 0.12)", border: "rgba(149, 236, 105, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #95ec69, #4ab724)", avatarText: "#101316" },
    "P4": { bg: "rgba(186, 104, 200, 0.12)", border: "rgba(186, 104, 200, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #ba68c8, #8e24aa)", avatarText: "#ffffff" },
    "P5": { bg: "rgba(255, 152, 0, 0.12)", border: "rgba(255, 152, 0, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #ff9800, #f57c00)", avatarText: "#ffffff" },
    "spk_1": { bg: "rgba(74, 163, 255, 0.12)", border: "rgba(74, 163, 255, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #4aa3ff, #2f80ed)", avatarText: "#ffffff" },
    "spk_2": { bg: "rgba(240, 211, 78, 0.12)", border: "rgba(240, 211, 78, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #f0d34e, #d4af37)", avatarText: "#101316" },
    "spk_3": { bg: "rgba(149, 236, 105, 0.12)", border: "rgba(149, 236, 105, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #95ec69, #4ab724)", avatarText: "#101316" },
    "spk_4": { bg: "rgba(186, 104, 200, 0.12)", border: "rgba(186, 104, 200, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #ba68c8, #8e24aa)", avatarText: "#ffffff" },
    "offscreen": { bg: "rgba(142, 150, 158, 0.10)", border: "rgba(142, 150, 158, 0.35)", text: "#eef3f7", avatarBg: "#303842", avatarText: "#8e969e" },
    "ambiguous": { bg: "rgba(240, 211, 78, 0.10)", border: "rgba(240, 211, 78, 0.45)", text: "#eef3f7", avatarBg: "#4a4328", avatarText: "#f0d34e" },
    "overlap": { bg: "rgba(255, 110, 110, 0.12)", border: "rgba(255, 110, 110, 0.6)", text: "#eef3f7", avatarBg: "linear-gradient(135deg, #ff6e6e, #d32f2f)", avatarText: "#ffffff" }
  };

  let id = speakerId || "unknown";
  if (themes[id]) {
    return themes[id];
  }
  if (id.startsWith("I")) {
    const mapped = id.replace("I", "P");
    if (themes[mapped]) return themes[mapped];
  }

  // Stable theme for dynamically discovered speaker IDs.
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = id.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash % 360);
  return {
    bg: `hsla(${hue}, 60%, 45%, 0.12)`,
    border: `hsla(${hue}, 70%, 65%, 0.6)`,
    text: "#eef3f7",
    avatarBg: `linear-gradient(135deg, hsl(${hue}, 70%, 55%), hsl(${hue}, 80%, 40%))`,
    avatarText: "#ffffff"
  };
}

function getSpeakerSide(event, speakerId) {
  if (event.position === "right") {
    return "right";
  }
  if (event.position === "left") {
    return "left";
  }
  if (event.position === "center") {
    return "left";
  }
  // Separate different off-screen or unassigned speakers dynamically
  const spk = speakerId || "";
  if (spk.endsWith("2") || spk.endsWith("4") || spk === "right") {
    return "right";
  }
  return "left";
}

function addOrUpdateBubble(event) {
  if (!event.text) return;

  const utteranceId = event.utterance_id || `temp_${Date.now()}`;
  const isPartial = event.type === "asr_partial" || event.partial;

  const speakerId = event.speaker_id || (event.person_track_ids && event.person_track_ids.length ? `P${event.person_track_ids[0]}` : "offscreen");
  const side = getSpeakerSide(event, speakerId);
  const theme = getSpeakerTheme(speakerId);
  const avText = avatarText(event);

  let row = activeBubbles[utteranceId];
  let avatar, bubble, meta;

  if (!row) {
    row = document.createElement("div");
    row.className = `bubble-row ${side}`;
    
    avatar = document.createElement("div");
    avatar.className = "avatar";
    
    bubble = document.createElement("div");
    bubble.className = "bubble";
    
    meta = document.createElement("div");
    meta.className = "bubble-meta";
    
    bubble.appendChild(meta);
    
    if (side === "right") {
      row.appendChild(bubble);
      row.appendChild(avatar);
    } else {
      row.appendChild(avatar);
      row.appendChild(bubble);
    }
    
    chatList.appendChild(row);
    activeBubbles[utteranceId] = row;
  } else {
    avatar = row.querySelector(".avatar");
    bubble = row.querySelector(".bubble");
    meta = row.querySelector(".bubble-meta");
    
    row.className = `bubble-row ${side}`;
    if (side === "right") {
      if (row.firstElementChild === avatar) {
        row.removeChild(avatar);
        row.appendChild(avatar);
      }
    } else {
      if (row.firstElementChild === bubble) {
        row.removeChild(bubble);
        row.insertBefore(avatar, bubble);
      }
    }
  }

  // Update elements
  avatar.textContent = avText;
  avatar.style.background = theme.avatarBg;
  avatar.style.color = theme.avatarText;
  avatar.style.border = `1px solid ${theme.border}`;
  avatar.style.boxShadow = "0 2px 8px rgba(0, 0, 0, 0.4)";

  bubble.className = `bubble ${event.tentative || isPartial ? "tentative" : ""}`;
  bubble.style.backgroundColor = theme.bg;
  bubble.style.borderColor = theme.border;
  bubble.style.borderWidth = "1px";
  bubble.style.borderStyle = (event.tentative || isPartial) ? "dashed" : "solid";
  bubble.style.color = theme.text;
  bubble.style.boxShadow = (event.tentative || isPartial) ? "none" : `0 4px 12px ${theme.bg}`;

  const displayName = event.speaker_name || avText;
  const label = displayName ? `[${displayName}] ` : "";
  const bubbleText = `${label}${event.text}`;

  while (bubble.firstChild && bubble.firstChild !== meta) {
    bubble.removeChild(bubble.firstChild);
  }
  const textNode = document.createTextNode(bubbleText);
  bubble.insertBefore(textNode, meta);

  const posLabel = event.position || "unknown";
  const confPercent = Math.round((event.confidence || 0) * 100);
  const activePercent = activeSpeakerPercent(event);
  const voiceLabel = event.voice_spk_id ? ` · ${event.voice_spk_id}` : "";
  const reason = event.decision_reason || event.debug?.decision_reason || "";
  const reasonLabel = reason ? ` · ${reason}` : "";
  meta.textContent = `${posLabel} · fusion ${confPercent}% · speaking ${activePercent}%${voiceLabel}${reasonLabel}`;
  meta.style.color = theme.border;
  meta.style.opacity = "0.75";

  if (!row.style.animation && !row.dataset.animated) {
    row.dataset.animated = "true";
    row.style.opacity = "0";
    row.style.transform = "translateY(12px) scale(0.98)";
    row.style.transition = "all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)";
    requestAnimationFrame(() => {
      row.style.opacity = "1";
      row.style.transform = "translateY(0) scale(1)";
    });
  }

  chatList.scrollTop = chatList.scrollHeight;
}

function activeSpeakerPercent(event) {
  const ids = event.person_track_ids || [];
  if (!ids.length || !Array.isArray(event.tracks)) {
    return 0;
  }
  const selected = event.tracks.find((track) => ids.includes(track.person_track_id));
  return Math.round(((selected && selected.p_active) || 0) * 100);
}

function avatarText(event) {
  if (event.speaker_id) {
    if (event.speaker_id === "offscreen") return "外";
    if (event.speaker_id === "ambiguous") return "?";
    if (event.speaker_id === "overlap") return "多";
    if (event.speaker_id.startsWith("spk_")) {
      return event.speaker_id.replace("spk_", "V");
    }
    return event.speaker_id;
  }
  const ids = event.person_track_ids || [];
  if (event.position === "offscreen") return "外";
  if (event.position === "overlap") return "多";
  if (!ids.length) return "?";
  return `P${ids[0]}`;
}

clearChat.addEventListener("click", () => {
  chatList.replaceChildren();
  activeBubbles = {};
});

async function refreshSyncHealth() {
  if (!syncState) return;
  try {
    const response = await fetch(getBackendUrl("/v2/status"));
    if (!response.ok) throw new Error("status unavailable");
    const status = await response.json();
    const lag = Math.round(status.sync?.watermark_lag_ms || 0);
    const healthy = status.sync?.healthy !== false;
    syncState.textContent = healthy ? `sync: healthy ${lag}ms` : `sync: gated ${lag}ms`;
    syncState.className = healthy ? "sync-healthy" : "sync-gated";
  } catch (_error) {
    syncState.textContent = "sync: unavailable";
    syncState.className = "sync-gated";
  }
}

window.addEventListener("resize", resizeOverlay);
startVideoPreview();
connectEvents();
drawOverlay();
refreshSyncHealth();
setInterval(refreshSyncHealth, 2000);

// Face Registry Frontend Logic
const galleryList = document.querySelector("#galleryList");
const uploadTrigger = document.querySelector("#uploadTrigger");
const imageInput = document.querySelector("#imageInput");
const uploadZone = document.querySelector("#uploadZone");
const registerName = document.querySelector("#registerName");
const registerBackground = document.querySelector("#registerBackground");

async function loadFaceGallery() {
  if (!galleryList) return;
  try {
    const res = await fetch(getBackendUrl("/v1/gallery"));
    if (!res.ok) throw new Error("Failed to fetch gallery");
    const gallery = await res.json();
    galleryList.replaceChildren();
    
    Object.entries(gallery).forEach(([id, info]) => {
      const card = document.createElement("div");
      card.className = "gallery-card";
      card.dataset.id = id;
      
      const initials = info.name ? info.name.substring(0, 2) : "Face";
      const avatar = document.createElement("div");
      avatar.className = "card-avatar";
      avatar.textContent = initials;
      
      const details = document.createElement("div");
      details.className = "card-details";
      
      const nameInput = document.createElement("input");
      nameInput.type = "text";
      nameInput.className = "card-name";
      nameInput.value = info.name || "";
      nameInput.placeholder = "Name";
      nameInput.addEventListener("blur", () => updateFaceMetadata(id, nameInput.value, bgInput.value));
      nameInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") nameInput.blur();
      });
      
      const bgInput = document.createElement("input");
      bgInput.type = "text";
      bgInput.className = "card-bg";
      bgInput.value = info.background || "";
      bgInput.placeholder = "Background Info";
      bgInput.addEventListener("blur", () => updateFaceMetadata(id, nameInput.value, bgInput.value));
      bgInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") bgInput.blur();
      });
      
      details.appendChild(nameInput);
      details.appendChild(bgInput);
      
      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "card-delete";
      deleteBtn.textContent = "×";
      deleteBtn.addEventListener("click", () => deleteFace(id, card));
      
      card.appendChild(avatar);
      card.appendChild(details);
      card.appendChild(deleteBtn);
      
      galleryList.appendChild(card);
    });
  } catch (err) {
    console.warn("Face Registry error:", err);
  }
}

async function uploadFaceImage(base64Image) {
  try {
    const name = (registerName?.value || "").trim();
    const background = (registerBackground?.value || "").trim();
    if (!name) {
      alert("请先填写姓名");
      registerName?.focus();
      return;
    }
    const res = await fetch(getBackendUrl("/v1/gallery/upload"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        background,
        image: base64Image
      })
    });
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || "Upload failed");
    }
    await loadFaceGallery();
    if (registerName) registerName.value = "";
    if (registerBackground) registerBackground.value = "";
  } catch (err) {
    alert("Face registration failed: " + err.message);
  }
}

async function updateFaceMetadata(id, name, background) {
  try {
    await fetch(getBackendUrl("/v1/gallery/update"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: parseInt(id), name, background })
    });
  } catch (err) {
    console.warn("Update metadata failed:", err);
  }
}

async function deleteFace(id, cardElement) {
  if (!confirm("Are you sure you want to delete this registered face?")) return;
  try {
    const res = await fetch(getBackendUrl("/v1/gallery/delete"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: parseInt(id) })
    });
    if (res.ok) {
      cardElement.style.opacity = "0";
      cardElement.style.transform = "translateX(20px)";
      setTimeout(() => cardElement.remove(), 250);
    }
  } catch (err) {
    console.warn("Delete face failed:", err);
  }
}

// Bind upload handlers
if (uploadTrigger && imageInput) {
  uploadTrigger.addEventListener("click", () => imageInput.click());
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => uploadFaceImage(evt.target.result);
    reader.readAsDataURL(file);
  });
}

// Drag & Drop
if (uploadZone) {
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (evt) => uploadFaceImage(evt.target.result);
      reader.readAsDataURL(file);
    }
  });
}

// Initial Load
loadFaceGallery();
