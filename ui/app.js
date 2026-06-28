const els = {
  status: document.querySelector("#status"),
  prompt: document.querySelector("#prompt"),
  model: document.querySelector("#model"),
  temperature: document.querySelector("#temperature"),
  temperatureValue: document.querySelector("#temperatureValue"),
  topK: document.querySelector("#topK"),
  topP: document.querySelector("#topP"),
  topPValue: document.querySelector("#topPValue"),
  maxNewTokens: document.querySelector("#maxNewTokens"),
  conversationalMode: document.querySelector("#conversationalMode"),
  dynamicContext: document.querySelector("#dynamicContext"),
  maxContext: document.querySelector("#maxContext"),
  generate: document.querySelector("#generate"),
  clear: document.querySelector("#clear"),
  output: document.querySelector("#output"),
  speed: document.querySelector("#speed"),
  metrics: document.querySelector("#metrics"),
  models: document.querySelector("#models"),
  agents: document.querySelector("#agents"),
  plugins: document.querySelector("#plugins"),
  refreshMetrics: document.querySelector("#refreshMetrics"),
};

let socket = null;
let streamBuffer = "";
let rafPending = false;

function setStatus(text) {
  els.status.textContent = text;
}

function payload() {
  const prompt = els.conversationalMode.checked && !els.prompt.value.includes("User:")
    ? `User: ${els.prompt.value}\nAssistant:`
    : els.prompt.value;
  return {
    prompt,
    model: els.model.value,
    temperature: Number(els.temperature.value),
    top_k: Number(els.topK.value),
    top_p: Number(els.topP.value),
    max_new_tokens: Number(els.maxNewTokens.value),
    dynamic_context: els.dynamicContext.checked,
    max_context: Number(els.maxContext.value),
  };
}

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/stream`;
}

function generate() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }
  els.output.textContent = "";
  streamBuffer = "";
  setStatus("Connecting");
  socket = new WebSocket(wsUrl());

  socket.addEventListener("open", () => {
    setStatus("Generating");
    socket.send(JSON.stringify(payload()));
  });

  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "token") {
      appendToken(data.text);
    } else if (data.type === "status") {
      setStatus(data.message);
    } else if (data.type === "metrics") {
      renderMetrics(data.metrics);
    } else if (data.type === "done") {
      flushStreamBuffer();
      setStatus("Done");
      els.speed.textContent = `${data.tokens_per_second.toFixed(2)} tok/s`;
      refreshMetrics();
      socket.close();
    } else if (data.type === "error") {
      setStatus(`Error: ${data.message}`);
      fallbackGenerate();
    }
  });

  socket.addEventListener("error", () => {
    setStatus("WebSocket error, using HTTP fallback");
    fallbackGenerate();
  });
}

function appendToken(text) {
  streamBuffer += text;
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    flushStreamBuffer();
    rafPending = false;
  });
}

function flushStreamBuffer() {
  if (!streamBuffer) return;
  els.output.textContent += streamBuffer;
  streamBuffer = "";
  els.output.scrollTop = els.output.scrollHeight;
}

async function fallbackGenerate() {
  try {
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    const data = await res.json();
    if (data.text) {
      els.output.textContent = data.text;
      setStatus("Done via fallback");
    } else {
      setStatus(data.error || "Fallback failed");
    }
  } catch (err) {
    setStatus(`Fallback error: ${err.message}`);
  }
}

function bytes(value) {
  if (value === null || value === undefined) return "n/a";
  const units = ["B", "KB", "MB", "GB"];
  let n = Number(value);
  for (const unit of units) {
    if (n < 1024) return `${n.toFixed(1)} ${unit}`;
    n /= 1024;
  }
  return `${n.toFixed(1)} TB`;
}

async function refreshMetrics() {
  const res = await fetch("/metrics");
  const data = await res.json();
  renderMetrics(data);
}

function renderMetrics(data) {
  els.metrics.innerHTML = "";
  const rows = [
    ["Tokens/sec", data.tokens_per_second?.toFixed?.(2) || "0"],
    ["Latency", `${(data.latency_ms || 0).toFixed(2)} ms/token`],
    ["RAM free", bytes(data.ram?.free)],
    ["RAM used", data.ram?.used_percent !== null ? `${data.ram.used_percent}%` : "n/a"],
    ["VRAM free", bytes(data.vram?.free)],
    ["VRAM used", data.vram?.used_percent !== null ? `${data.vram.used_percent}%` : "n/a"],
    ["VRAM allocated", bytes(data.vram?.allocated)],
    ["Last error", data.last_error || "none"],
  ];
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    els.metrics.append(dt, dd);
  }
}

async function refreshModels() {
  const res = await fetch("/models");
  const data = await res.json();
  els.models.innerHTML = "";
  for (const model of data.models || []) {
    const li = document.createElement("li");
    li.textContent = `${model.label} ${model.available ? "available" : "unavailable"}`;
    li.className = model.available ? "ok" : "muted";
    els.models.append(li);
  }
}

async function refreshList(path, target, emptyText) {
  const res = await fetch(path);
  const data = await res.json();
  const items = data.agents || data.plugins || [];
  target.innerHTML = "";
  if (items.length === 0) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    li.className = "muted";
    target.append(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item.name;
    target.append(li);
  }
}

els.temperature.addEventListener("input", () => {
  els.temperatureValue.textContent = els.temperature.value;
});
els.topP.addEventListener("input", () => {
  els.topPValue.textContent = els.topP.value;
});
els.generate.addEventListener("click", generate);
els.clear.addEventListener("click", () => {
  els.output.textContent = "";
});
els.refreshMetrics.addEventListener("click", refreshMetrics);

refreshMetrics();
refreshModels();
refreshList("/agents", els.agents, "No agents found");
refreshList("/plugins", els.plugins, "No plugins found");
setInterval(refreshMetrics, 3000);
