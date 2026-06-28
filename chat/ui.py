CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MiniLLM Chat Mode</title>
  <style>
    :root { color-scheme: dark; --bg:#0b0d10; --panel:#14181d; --line:#252b33; --text:#edf1f5; --muted:#8d99a8; --user:#1e3a5f; --assistant:#18251f; --accent:#5aa9ff; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:var(--bg); color:var(--text); font:14px/1.45 Inter, ui-sans-serif, system-ui, Segoe UI, Arial; display:grid; grid-template-rows:auto 1fr auto; }
    header { padding:14px 18px; border-bottom:1px solid var(--line); background:#101317; display:flex; align-items:center; justify-content:space-between; gap:12px; }
    h1 { margin:0; font-size:17px; letter-spacing:0; }
    .status { color:var(--muted); font-size:12px; }
    main { overflow:auto; padding:18px; display:flex; flex-direction:column; gap:12px; }
    .bubble { max-width:860px; border:1px solid var(--line); border-radius:8px; padding:12px 14px; white-space:pre-wrap; overflow-wrap:anywhere; }
    .user { align-self:flex-end; background:var(--user); }
    .assistant { align-self:flex-start; background:var(--assistant); }
    .role { color:var(--muted); font-size:12px; margin-bottom:4px; }
    footer { border-top:1px solid var(--line); background:#101317; padding:12px; display:grid; gap:10px; }
    .controls { display:grid; grid-template-columns: minmax(180px, 1fr) repeat(3, minmax(90px, 130px)); gap:8px; }
    .composer { display:grid; grid-template-columns:1fr auto; gap:8px; }
    input, select, textarea, button { border:1px solid var(--line); background:#0f1216; color:var(--text); border-radius:6px; padding:9px 10px; font:inherit; }
    textarea { min-height:54px; max-height:160px; resize:vertical; }
    button { cursor:pointer; background:#1b222b; font-weight:650; }
    button:hover { border-color:#3b4654; }
    button.primary { background:#17304f; color:#d9ecff; min-width:92px; }
    @media (max-width:760px) { .controls { grid-template-columns:1fr 1fr; } .composer { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>MiniLLM Chat Mode</h1>
    <div class="status" id="status">professional local mode</div>
  </header>
  <main id="chat"></main>
  <footer>
    <div class="controls">
      <select id="checkpoint"></select>
      <input id="temperature" type="number" min="0.1" max="1.2" step="0.05" value="0.45" title="temperature" />
      <input id="topP" type="number" min="0.1" max="1" step="0.01" value="0.82" title="top_p" />
      <input id="maxTokens" type="number" min="1" max="160" step="1" value="80" title="max_tokens" />
    </div>
    <div class="composer">
      <textarea id="prompt" placeholder="Scrivi un messaggio a MiniLLM..."></textarea>
      <button class="primary" id="send">Send</button>
    </div>
  </footer>
  <script>
    let history = [];
    const chat = document.getElementById('chat');
    const statusEl = document.getElementById('status');
    function addBubble(role, content='') {
      const el = document.createElement('div');
      el.className = `bubble ${role}`;
      el.innerHTML = `<div class="role">${role === 'user' ? 'User' : 'Assistant'}</div><span></span>`;
      el.querySelector('span').textContent = content;
      chat.appendChild(el);
      chat.scrollTop = chat.scrollHeight;
      return el.querySelector('span');
    }
    async function loadCheckpoints() {
      const res = await fetch('/api/chat/checkpoints');
      const data = await res.json();
      const select = document.getElementById('checkpoint');
      select.innerHTML = data.checkpoints.map(c => `<option value="${c.path}" ${c.active ? 'selected' : ''}>${c.name}</option>`).join('');
      statusEl.textContent = `checkpoint: ${data.active_checkpoint || 'none'}`;
    }
    async function sendMessage() {
      const promptEl = document.getElementById('prompt');
      const prompt = promptEl.value.trim();
      if (!prompt) return;
      promptEl.value = '';
      addBubble('user', prompt);
      const assistantSpan = addBubble('assistant', '');
      statusEl.textContent = 'generating...';
      const payload = {
        prompt,
        history,
        checkpoint_path: document.getElementById('checkpoint').value,
        temperature: Number(document.getElementById('temperature').value),
        top_p: Number(document.getElementById('topP').value),
        max_tokens: Number(document.getElementById('maxTokens').value),
        stream: true
      };
      const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let text = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, {stream:true});
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (!data || data === '[DONE]') continue;
          const event = JSON.parse(data);
          if (event.token) {
            text += event.token;
            assistantSpan.textContent = text;
            chat.scrollTop = chat.scrollHeight;
          }
          if (event.history) history = event.history;
        }
      }
      statusEl.textContent = `checkpoint: ${document.getElementById('checkpoint').value}`;
    }
    document.getElementById('send').addEventListener('click', sendMessage);
    document.getElementById('prompt').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    loadCheckpoints();
  </script>
</body>
</html>"""
