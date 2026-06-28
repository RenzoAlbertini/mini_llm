import argparse
import csv
import json
import math
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from training.control import (
    is_process_alive,
    read_control,
    read_pid,
    stop_process,
    write_control,
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MiniLLM Training Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#0b0d10; --panel:#14181d; --line:#252b33; --text:#edf1f5; --muted:#8d99a8; --green:#49d17d; --blue:#5aa9ff; --red:#ff6b6b; --amber:#f5b84b; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 Inter, ui-sans-serif, system-ui, Segoe UI, Arial; }
    header { display:flex; align-items:center; justify-content:space-between; padding:18px 22px; border-bottom:1px solid var(--line); background:#101317; position:sticky; top:0; z-index:2; }
    h1 { margin:0; font-size:18px; letter-spacing:0; }
    main { padding:18px; display:grid; gap:16px; grid-template-columns: 1.25fr .75fr; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }
    .grid { display:grid; gap:12px; grid-template-columns:repeat(2,minmax(0,1fr)); }
    .kpis { display:grid; gap:10px; grid-template-columns:repeat(4,minmax(0,1fr)); }
    .kpi { border:1px solid var(--line); border-radius:6px; padding:10px; background:#101317; min-height:70px; }
    .label { color:var(--muted); font-size:12px; }
    .value { margin-top:5px; font-size:20px; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    canvas { width:100%; height:260px; display:block; background:#0f1216; border:1px solid var(--line); border-radius:6px; }
    h2 { margin:0 0 10px; font-size:14px; }
    pre { height:320px; overflow:auto; margin:0; padding:12px; border:1px solid var(--line); border-radius:6px; background:#0f1216; color:#d7dde6; white-space:pre-wrap; }
    button { border:1px solid var(--line); background:#1b222b; color:var(--text); padding:9px 12px; border-radius:6px; cursor:pointer; font-weight:650; }
    button:hover { border-color:#3b4654; }
    button.stop { background:#34191b; color:#ffd5d5; }
    button.pause { background:#342a17; color:#ffe2a8; }
    button.resume { background:#17301f; color:#c8ffd8; }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .status { color:var(--muted); }
    .side { display:grid; gap:16px; align-content:start; }
    table { width:100%; border-collapse:collapse; }
    td { padding:7px 0; border-bottom:1px solid var(--line); vertical-align:top; }
    td:first-child { color:var(--muted); width:42%; }
    @media (max-width: 980px) { main { grid-template-columns:1fr; } .grid,.kpis { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>MiniLLM Training Dashboard</h1>
    <div class="status" id="heartbeat">connecting</div>
  </header>
  <main>
    <div>
      <section>
        <div class="kpis">
          <div class="kpi"><div class="label">GPU Temp</div><div class="value" id="gpuTemp">--</div></div>
          <div class="kpi"><div class="label">GPU Util</div><div class="value" id="gpuUtil">--</div></div>
          <div class="kpi"><div class="label">VRAM</div><div class="value" id="vram">--</div></div>
          <div class="kpi"><div class="label">Current Epoch</div><div class="value" id="epoch">--</div></div>
        </div>
      </section>
      <section>
        <h2>Plots</h2>
        <div class="grid">
          <canvas id="lossStep"></canvas>
          <canvas id="lossEpoch"></canvas>
          <canvas id="gpuTempChart"></canvas>
          <canvas id="gpuUtilChart"></canvas>
          <canvas id="vramChart"></canvas>
          <canvas id="benchPplChart"></canvas>
          <canvas id="benchCoherenceChart"></canvas>
          <canvas id="benchRepetitionChart"></canvas>
        </div>
        <div class="row" style="margin-top:12px">
          <button onclick="savePlots()">Save PNG Plots</button>
          <span class="status" id="plotStatus"></span>
        </div>
      </section>
      <section>
        <h2>Training Log</h2>
        <pre id="log"></pre>
      </section>
    </div>
    <div class="side">
      <section>
        <h2>Model Status</h2>
        <table>
          <tr><td>Training PID</td><td id="pid">--</td></tr>
          <tr><td>Last checkpoint</td><td id="checkpoint">--</td></tr>
          <tr><td>Model size</td><td id="modelSize">--</td></tr>
          <tr><td>seq_len</td><td id="seqLen">--</td></tr>
          <tr><td>batch_size</td><td id="batchSize">--</td></tr>
          <tr><td>Step</td><td id="step">--</td></tr>
          <tr><td>Control</td><td id="control">--</td></tr>
        </table>
      </section>
      <section>
        <h2>Training Controls</h2>
        <div class="row">
          <button class="pause" onclick="control('pause')">Pause Training</button>
          <button class="resume" onclick="control('resume')">Resume Training</button>
          <button class="stop" onclick="control('stop')">Stop Training</button>
        </div>
        <p class="status" id="controlStatus"></p>
      </section>
      <section>
        <h2>Model Evaluation</h2>
        <div class="row">
          <input id="benchmarkCheckpoint" style="flex:1; min-width:180px; background:#0f1216; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:9px" placeholder="models/checkpoints/best.pt" />
          <button onclick="runBenchmark()">Run Benchmark</button>
        </div>
        <p class="status" id="benchmarkStatus"></p>
        <table id="benchmarkResults"></table>
      </section>
      <section>
        <h2>Chat Mode</h2>
        <div class="row">
          <button onclick="openChat()">Open Chat Mode</button>
          <span class="status" id="chatCheckpoint">checkpoint: --</span>
        </div>
      </section>
      <section>
        <h2>Checkpoints</h2>
        <table id="checkpoints"></table>
      </section>
    </div>
  </main>
  <script>
    const charts = {};
    function fmt(v, unit='') { return v === null || v === undefined || Number.isNaN(v) ? '--' : `${v}${unit}`; }
    function drawChart(id, title, points, color) {
      const canvas = document.getElementById(id), dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect(); canvas.width = rect.width*dpr; canvas.height = rect.height*dpr;
      const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
      const w = rect.width, h = rect.height, p = 34;
      ctx.clearRect(0,0,w,h); ctx.fillStyle='#0f1216'; ctx.fillRect(0,0,w,h);
      ctx.strokeStyle='#252b33'; ctx.lineWidth=1; ctx.strokeRect(.5,.5,w-1,h-1);
      ctx.fillStyle='#edf1f5'; ctx.font='12px system-ui'; ctx.fillText(title, p, 22);
      if (!points.length) return;
      const xs = points.map(p=>p.x), ys = points.map(p=>p.y).filter(Number.isFinite);
      if (!ys.length) return;
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const sx = x => p + ((x-minX)/Math.max(1e-9,maxX-minX))*(w-p*1.5);
      const sy = y => h-p - ((y-minY)/Math.max(1e-9,maxY-minY))*(h-p*2);
      ctx.strokeStyle='#2a313b'; for (let i=0;i<4;i++){ const y=p+i*(h-p*2)/3; ctx.beginPath(); ctx.moveTo(p,y); ctx.lineTo(w-p/2,y); ctx.stroke(); }
      ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); let started=false;
      for (const pt of points) { if (!Number.isFinite(pt.y)) continue; const x=sx(pt.x), y=sy(pt.y); if(!started){ctx.moveTo(x,y); started=true;} else ctx.lineTo(x,y); }
      ctx.stroke();
      ctx.fillStyle='#8d99a8'; ctx.fillText(minY.toFixed(2), 6, h-p); ctx.fillText(maxY.toFixed(2), 6, p+5);
    }
    async function refresh() {
      const res = await fetch('/api/state'); const s = await res.json();
      document.getElementById('heartbeat').textContent = new Date().toLocaleTimeString();
      const g = s.gpu.latest || {};
      document.getElementById('gpuTemp').textContent = fmt(g.temperature_gpu,' C');
      document.getElementById('gpuUtil').textContent = fmt(g.utilization_gpu,'%');
      document.getElementById('vram').textContent = g.memory_used_mb ? `${g.memory_used_mb}/${g.memory_total_mb} MB` : '--';
      document.getElementById('epoch').textContent = fmt(s.status.epoch);
      document.getElementById('pid').textContent = s.status.training_pid ? `${s.status.training_pid} (${s.status.process_alive ? 'running':'stopped'})` : '--';
      document.getElementById('checkpoint').textContent = s.status.latest_checkpoint || '--';
      document.getElementById('modelSize').textContent = s.status.model_size || '--';
      document.getElementById('seqLen').textContent = fmt(s.status.seq_len);
      document.getElementById('batchSize').textContent = fmt(s.status.batch_size);
      document.getElementById('step').textContent = fmt(s.status.step);
      document.getElementById('control').textContent = s.control.paused ? 'paused' : 'running';
      document.getElementById('log').textContent = s.log_tail || '';
      const ckpt = document.getElementById('checkpoints');
      ckpt.innerHTML = s.checkpoints.map(c => `<tr><td>${c.name}</td><td>${c.size_mb} MB<br><span class="status">${c.modified}</span></td></tr>`).join('');
      drawChart('lossStep','loss vs step', s.series.loss_step, '#49d17d');
      drawChart('lossEpoch','loss vs epoch', s.series.loss_epoch, '#5aa9ff');
      drawChart('gpuTempChart','GPU temperature', s.series.gpu_temp, '#ff6b6b');
      drawChart('gpuUtilChart','GPU utilization', s.series.gpu_util, '#f5b84b');
      drawChart('vramChart','VRAM usage', s.series.vram, '#9b8cff');
      drawChart('benchPplChart','benchmark perplexity vs checkpoint', s.series.benchmark_perplexity, '#49d17d');
      drawChart('benchCoherenceChart','benchmark coherence vs checkpoint', s.series.benchmark_coherence, '#5aa9ff');
      drawChart('benchRepetitionChart','benchmark repetition score vs checkpoint', s.series.benchmark_repetition, '#f5b84b');
      const benchInput = document.getElementById('benchmarkCheckpoint');
      if (!benchInput.value && s.status.latest_checkpoint) benchInput.value = s.status.latest_checkpoint;
      const bench = document.getElementById('benchmarkResults');
      bench.innerHTML = (s.benchmarks || []).map((r, idx) => `<tr><td>${r.checkpoint_name || idx}</td><td>ppl ${fmt(r.metrics.perplexity?.toFixed ? r.metrics.perplexity.toFixed(2) : r.metrics.perplexity)}<br>coh ${fmt(r.metrics.response_coherence_score?.toFixed ? r.metrics.response_coherence_score.toFixed(3) : r.metrics.response_coherence_score)}<br>rep ${fmt(r.metrics.repetition_penalty_score?.toFixed ? r.metrics.repetition_penalty_score.toFixed(3) : r.metrics.repetition_penalty_score)}</td></tr>`).join('');
      document.getElementById('chatCheckpoint').textContent = `checkpoint: ${s.chat?.active_checkpoint || 'chat server offline'}`;
    }
    async function control(action) {
      const res = await fetch(`/api/control/${action}`, {method:'POST'});
      document.getElementById('controlStatus').textContent = JSON.stringify(await res.json());
      refresh();
    }
    async function savePlots() {
      const res = await fetch('/api/plots/save', {method:'POST'});
      const data = await res.json();
      document.getElementById('plotStatus').textContent = data.files.join(', ');
    }
    async function runBenchmark() {
      const checkpoint = document.getElementById('benchmarkCheckpoint').value || 'models/checkpoints/best.pt';
      const status = document.getElementById('benchmarkStatus');
      status.textContent = 'running benchmark...';
      const res = await fetch('/api/evaluate', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({checkpoint_path: checkpoint})
      });
      const data = await res.json();
      status.textContent = data.result_path ? `saved ${data.result_path}` : JSON.stringify(data);
      refresh();
    }
    function openChat() {
      window.open('/chat-mode', '_blank');
    }
    refresh(); setInterval(refresh, 2000); window.addEventListener('resize', refresh);
  </script>
</body>
</html>"""


class GpuMonitor:
    def __init__(self, out_path="data/logs/gpu_stats.csv", interval=2.0, max_samples=1800):
        self.out_path = Path(out_path)
        self.interval = interval
        self.max_samples = max_samples
        self.samples = []
        self.lock = threading.Lock()
        self.thread = None
        self.running = False

    def start(self):
        if self.running:
            return
        self.running = True
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            sample = query_gpu()
            if sample:
                with self.lock:
                    self.samples.append(sample)
                    self.samples = self.samples[-self.max_samples :]
                append_gpu_csv(self.out_path, sample)
            time.sleep(self.interval)

    def snapshot(self):
        with self.lock:
            return list(self.samples)


def query_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        return None
    try:
        return {
            "time": time.time(),
            "utilization_gpu": int(float(parts[0])),
            "temperature_gpu": int(float(parts[1])),
            "memory_used_mb": int(float(parts[2])),
            "memory_total_mb": int(float(parts[3])),
            "power_w": float(parts[4]),
        }
    except ValueError:
        return None


def append_gpu_csv(path, sample):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["time", "utilization_gpu", "temperature_gpu", "memory_used_mb", "memory_total_mb", "power_w"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(sample)


def read_stats(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_log_tail(path, max_chars=12000):
    path = Path(path)
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="ignore")
    return data[-max_chars:]


def load_run_config(checkpoint_dir):
    path = Path(checkpoint_dir) / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def estimate_params(config):
    if not config:
        return None
    vocab = int(config.get("vocab_size", 0))
    seq = int(config.get("seq_len", 0))
    d = int(config.get("d_model", config.get("hidden_size", 0)))
    layers = int(config.get("n_layers", 0))
    d_ff = int(config.get("d_ff", 0))
    bias = 1 if config.get("bias", True) else 0
    embeddings = vocab * d + seq * d
    block = 3 * d * d + 3 * d * bias + d * d + d * bias + d * d_ff + d_ff * bias + d_ff * d + d * bias + 4 * d
    final_ln = 2 * d
    return embeddings + layers * block + final_ln


def list_checkpoints(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    rows = []
    for path in sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "mtime": stat.st_mtime,
            }
        )
    return rows


def find_gpu_python_pid():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and "python" in parts[1].lower():
            try:
                return int(parts[0])
            except ValueError:
                pass
    return None


def current_status(stats, checkpoints, checkpoint_dir):
    config_payload = load_run_config(checkpoint_dir)
    model_config = config_payload.get("model", {})
    training_config = config_payload.get("training", {})
    last_row = stats[-1] if stats else {}
    latest_checkpoint = checkpoints[0]["path"] if checkpoints else None
    params = estimate_params(model_config)
    pid = read_pid() or find_gpu_python_pid()
    return {
        "training_pid": pid,
        "process_alive": is_process_alive(pid) if pid else False,
        "latest_checkpoint": latest_checkpoint,
        "model_size": f"{params / 1e6:.2f}M params" if params else None,
        "seq_len": model_config.get("seq_len"),
        "batch_size": training_config.get("batch_size"),
        "epoch": int(float(last_row["epoch"])) if last_row.get("epoch") else None,
        "step": int(float(last_row["step"])) if last_row.get("step") else None,
    }


def build_series(stats, gpu_samples):
    loss_step = []
    by_epoch = {}
    for row in stats:
        try:
            step = float(row["step"])
            epoch = float(row["epoch"])
            loss = row.get("train_loss")
            val_loss = row.get("val_loss")
            y = float(loss or val_loss)
        except (ValueError, TypeError):
            continue
        loss_step.append({"x": step, "y": y})
        by_epoch.setdefault(epoch, []).append(y)
    loss_epoch = [{"x": epoch, "y": sum(values) / len(values)} for epoch, values in sorted(by_epoch.items())]
    gpu_temp = [{"x": s["time"], "y": s["temperature_gpu"]} for s in gpu_samples]
    gpu_util = [{"x": s["time"], "y": s["utilization_gpu"]} for s in gpu_samples]
    vram = [{"x": s["time"], "y": s["memory_used_mb"]} for s in gpu_samples]
    return {
        "loss_step": loss_step[-1000:],
        "loss_epoch": loss_epoch[-200:],
        "gpu_temp": gpu_temp[-1000:],
        "gpu_util": gpu_util[-1000:],
        "vram": vram[-1000:],
    }


def list_benchmark_results(benchmark_dir):
    benchmark_dir = Path(benchmark_dir)
    if not benchmark_dir.exists():
        return []
    rows = []
    for path in sorted(benchmark_dir.glob("results_*.json"), key=lambda p: p.stat().st_mtime):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["result_path"] = str(path)
        payload["mtime"] = path.stat().st_mtime
        rows.append(payload)
    return rows


def add_benchmark_series(series, benchmark_results):
    ppl = []
    coherence = []
    repetition = []
    for idx, row in enumerate(benchmark_results, start=1):
        metrics = row.get("metrics", {})
        if metrics.get("perplexity") is not None:
            ppl.append({"x": idx, "y": float(metrics["perplexity"])})
        if metrics.get("response_coherence_score") is not None:
            coherence.append({"x": idx, "y": float(metrics["response_coherence_score"])})
        if metrics.get("repetition_penalty_score") is not None:
            repetition.append({"x": idx, "y": float(metrics["repetition_penalty_score"])})
    series["benchmark_perplexity"] = ppl
    series["benchmark_coherence"] = coherence
    series["benchmark_repetition"] = repetition
    return series


def make_png(path, points, width=960, height=420, color=(73, 209, 125)):
    pixels = bytearray([15, 18, 22] * width * height)
    def put(x, y, rgb):
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pixels[i:i + 3] = bytes(rgb)
    def line(x0, y0, x1, y1, rgb):
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    put(x0 + ox, y0 + oy, rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
    pad = 36
    for x in range(pad, width - pad):
        put(x, height - pad, (42, 49, 59))
    for y in range(pad, height - pad):
        put(pad, y, (42, 49, 59))
    values = [(float(p["x"]), float(p["y"])) for p in points if p.get("y") is not None and math.isfinite(float(p["y"]))]
    if len(values) >= 2:
        xs, ys = [p[0] for p in values], [p[1] for p in values]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        def sx(x): return int(pad + (x - min_x) / max(1e-9, max_x - min_x) * (width - pad * 2))
        def sy(y): return int(height - pad - (y - min_y) / max(1e-9, max_y - min_y) * (height - pad * 2))
        for (x0, y0), (x1, y1) in zip(values, values[1:]):
            line(sx(x0), sy(y0), sx(x1), sy(y1), color)
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    def chunk(kind, data):
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00") + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def create_app(args):
    monitor = GpuMonitor(args.gpu_stats, interval=args.gpu_interval)
    monitor.start()
    app = FastAPI(title="MiniLLM Training Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/chat-mode")
    def chat_mode():
        return RedirectResponse(args.chat_url)

    @app.get("/api/chat/checkpoints")
    def chat_checkpoints():
        api_base = args.chat_url.rstrip("/")
        if api_base.endswith("/chat"):
            api_base = api_base[:-5]
        try:
            with urllib.request.urlopen(f"{api_base}/api/chat/checkpoints", timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return {"active_checkpoint": None, "checkpoints": [], "error": str(exc)}

    @app.get("/api/state")
    def state():
        stats = read_stats(args.stats_path)
        checkpoints = list_checkpoints(args.checkpoint_dir)
        gpu_samples = monitor.snapshot()
        benchmarks = list_benchmark_results(args.benchmark_dir)
        series = add_benchmark_series(build_series(stats, gpu_samples), benchmarks)
        return {
            "status": current_status(stats, checkpoints, args.checkpoint_dir),
            "control": read_control(),
            "checkpoints": checkpoints[:12],
            "benchmarks": benchmarks[-12:],
            "chat": chat_checkpoints(),
            "log_tail": read_log_tail(args.log_path),
            "gpu": {"latest": gpu_samples[-1] if gpu_samples else None},
            "series": series,
        }

    @app.post("/api/control/pause")
    def pause():
        return write_control(paused=True, stop_requested=False)

    @app.post("/api/control/resume")
    def resume():
        return write_control(paused=False, stop_requested=False)

    @app.post("/api/control/stop")
    def stop():
        pid = read_pid() or find_gpu_python_pid()
        write_control(stop_requested=True)
        ok, message = stop_process(pid, force=False) if pid else (False, "nessun processo training rilevato")
        return {"ok": ok, "pid": pid, "message": message}

    @app.post("/api/plots/save")
    def save_plots():
        stats = read_stats(args.stats_path)
        benchmarks = list_benchmark_results(args.benchmark_dir)
        series = add_benchmark_series(build_series(stats, monitor.snapshot()), benchmarks)
        out_dir = Path(args.plots_dir)
        specs = {
            "loss_vs_step.png": (series["loss_step"], (73, 209, 125)),
            "loss_vs_epoch.png": (series["loss_epoch"], (90, 169, 255)),
            "gpu_temperature.png": (series["gpu_temp"], (255, 107, 107)),
            "gpu_utilization.png": (series["gpu_util"], (245, 184, 75)),
            "vram_usage.png": (series["vram"], (155, 140, 255)),
            "benchmark_perplexity.png": (series["benchmark_perplexity"], (73, 209, 125)),
            "benchmark_coherence.png": (series["benchmark_coherence"], (90, 169, 255)),
            "benchmark_repetition.png": (series["benchmark_repetition"], (245, 184, 75)),
        }
        files = []
        for name, (points, color) in specs.items():
            path = out_dir / name
            make_png(path, points, color=color)
            files.append(str(path))
        return {"files": files}

    @app.post("/api/evaluate")
    def evaluate(payload: dict):
        checkpoint_path = payload.get("checkpoint_path") or payload.get("checkpoint") or "models/checkpoints/best.pt"
        from benchmark.evaluate import evaluate_checkpoint

        result = evaluate_checkpoint(
            checkpoint=checkpoint_path,
            dataset_path=payload.get("dataset_path", args.benchmark_dataset),
            tokenizer_path=payload.get("tokenizer_path", args.tokenizer),
            out_dir=args.benchmark_dir,
            device_name=payload.get("device", "auto"),
            max_samples=int(payload.get("max_samples", 0) or 0),
            max_new_tokens=int(payload.get("max_new_tokens", 64) or 64),
        )
        return result

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="MiniLLM local training dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--stats_path", default="data/logs/training_stats.csv")
    parser.add_argument("--log_path", default="data/logs/training.log")
    parser.add_argument("--checkpoint_dir", default="models/checkpoints")
    parser.add_argument("--plots_dir", default="data/plots")
    parser.add_argument("--gpu_stats", default="data/logs/gpu_stats.csv")
    parser.add_argument("--gpu_interval", type=float, default=2.0)
    parser.add_argument("--benchmark_dataset", default="benchmark/dataset.jsonl")
    parser.add_argument("--benchmark_dir", default="data/benchmarks")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--chat_url", default="http://127.0.0.1:8020/chat")
    return parser.parse_args()


DEFAULT_ARGS = parse_args() if __name__ == "__main__" else argparse.Namespace(
    host="127.0.0.1",
    port=8010,
    stats_path="data/logs/training_stats.csv",
    log_path="data/logs/training.log",
    checkpoint_dir="models/checkpoints",
    plots_dir="data/plots",
    gpu_stats="data/logs/gpu_stats.csv",
    gpu_interval=2.0,
    benchmark_dataset="benchmark/dataset.jsonl",
    benchmark_dir="data/benchmarks",
    tokenizer="tokenizer/tokenizer.json",
    chat_url="http://127.0.0.1:8020/chat",
)
app = create_app(DEFAULT_ARGS)


def main():
    args = DEFAULT_ARGS
    import uvicorn

    print(f"MiniLLM dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
