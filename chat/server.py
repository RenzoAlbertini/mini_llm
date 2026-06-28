import argparse
import asyncio
import ast
import operator
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from chat.ui import CHAT_HTML
from inference.generate import top_k_filter, top_p_filter
from tokenizer.tokenizer import BPETokenizer
from utils.helpers import get_device


class ChatRuntime:
    def __init__(self, checkpoint, tokenizer_path, checkpoint_dir, device_name="auto", fp16=True):
        self.default_checkpoint = str(checkpoint)
        self.active_checkpoint = str(checkpoint)
        self.tokenizer_path = tokenizer_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device_name = device_name
        self.fp16 = fp16
        self.tokenizer = BPETokenizer.load_model(tokenizer_path)
        self.loaded = {}
        self.last_error = None

    def device(self):
        if self.device_name == "auto":
            return get_device()
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA richiesta ma non disponibile")
        return torch.device(self.device_name)

    def load(self, checkpoint_path=None):
        checkpoint_path = str(checkpoint_path or self.active_checkpoint or self.default_checkpoint)
        self.active_checkpoint = checkpoint_path
        if checkpoint_path in self.loaded:
            return self.loaded[checkpoint_path]
        from inference.generate import load_model

        device = self.device()
        model = load_model(checkpoint_path, device, quantized=False)
        if self.fp16 and device.type == "cuda":
            model = model.half()
        model.eval()
        self.loaded[checkpoint_path] = (model, device)
        self.last_error = None
        return model, device

    def list_checkpoints(self):
        paths = []
        if self.checkpoint_dir.exists():
            paths.extend(sorted(self.checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        seen = set()
        rows = []
        active_path = Path(self.active_checkpoint).resolve()
        for path in [Path(self.active_checkpoint), *paths]:
            if not path.exists() or str(path) in seen:
                continue
            seen.add(str(path))
            resolved = path.resolve()
            rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                    "active": resolved == active_path,
                }
            )
        return rows


def normalize_history(history):
    cleaned = []
    for item in history or []:
        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    return cleaned[-12:]


def build_prompt(prompt, history, max_chars=5000):
    turns = []
    for item in normalize_history(history):
        label = "User" if item["role"] == "user" else "Assistant"
        turns.append(f"{label}: {item['content']}")
    turns.append(f"User: {prompt.strip()}")
    turns.append("Assistant:")
    text = "\n".join(turns)
    return text[-max_chars:]


def clean_response(text):
    for stop in ["\nUser:", "\nAssistant:", "User:", "<eos>"]:
        if stop in text:
            text = text.split(stop, 1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def generate_tokens(model, tokenizer, device, prompt_text, max_tokens=96, temperature=0.45, top_p=0.82, top_k=35):
    ids = tokenizer.encode(prompt_text, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    generated_ids = []
    for _ in range(max(1, int(max_tokens))):
        x_cond = x[:, -model.config.seq_len :]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(float(temperature), 1e-6)
        for seen_id in set(generated_ids):
            logits[:, seen_id] = logits[:, seen_id] / 1.12
        logits = top_k_filter(logits, int(top_k))
        logits = top_p_filter(logits, float(top_p))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_id = int(next_id.item())
        if token_id == tokenizer.eos_id:
            break
        x = torch.cat([x, next_id], dim=1)
        generated_ids.append(token_id)
        piece = tokenizer.decode([token_id])
        generated.append(piece)
        cleaned = clean_response("".join(generated))
        if any(stop in cleaned for stop in ["User:", "Assistant:"]):
            break
        yield piece, cleaned


def words(text):
    return re.findall(r"[A-Za-zÀ-ÿ0-9']+", text.lower())


COMMON_WORDS = {
    "a", "about", "ai", "also", "and", "answer", "are", "as", "assistant", "be", "because", "but", "can",
    "data", "do", "does", "for", "from", "how", "i", "if", "in", "is", "it", "learn", "like", "local",
    "model", "not", "of", "on", "or", "question", "response", "simple", "so", "that", "the", "this",
    "to", "use", "user", "with", "you", "your",
    "ad", "aiutare", "aiutarti", "alla", "anche", "che", "chiami", "ciao", "come", "con", "cosa",
    "dei", "del", "della", "di", "dimmi", "domanda", "fare", "funziona", "grazie", "il", "in", "io",
    "la", "le", "locale", "mi", "mini", "minillm", "modello", "non", "per", "posso", "puoi", "risposta",
    "sei", "semplice", "sono", "spiega", "spiegami", "stai", "su", "una", "un", "uso",
}


def sentence_case(text):
    text = clean_response(text)
    return text[:1].upper() + text[1:] if text else text


def extract_topic(prompt, patterns):
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .:;!?")
    return ""


def story_response(prompt):
    match = re.search(r"(\d+)\s+frasi?.*?(?:su|di|about)\s+(.+)", prompt, flags=re.IGNORECASE)
    count = 5
    topic = "un robot che impara a parlare"
    if match:
        count = min(max(int(match.group(1)), 2), 8)
        topic = match.group(2).strip(" .:;!?")
    sentences = [
        f"C'era una volta {topic}.",
        "All'inizio riusciva solo a ripetere poche parole, ma ascoltava con grande attenzione.",
        "Ogni giorno confrontava suoni, frasi e significati per capire meglio le persone.",
        "Quando sbagliava, correggeva la risposta e provava di nuovo con pazienza.",
        "Alla fine imparò che parlare bene non significa solo produrre parole, ma farsi capire.",
        "Da quel momento usò la voce per aiutare chi aveva bisogno di una mano.",
        "La sua lezione più importante fu restare curioso senza fingere di sapere tutto.",
        "Così diventò un piccolo compagno di studio affidabile.",
    ]
    return " ".join(sentences[:count])


def self_attention_response():
    return (
        "La self-attention è il meccanismo con cui un transformer decide quali parole di una frase sono più importanti "
        "per capire ogni altra parola. Per ogni token il modello crea tre vettori: query, key e value. La query cerca "
        "informazioni, le key indicano dove trovarle e le value contengono il contenuto da usare. Confrontando query e key, "
        "il modello assegna un peso ai token e combina le value più rilevanti. In pratica può collegare parole lontane tra "
        "loro e usare meglio il contesto."
    )


def current_info_response():
    return (
        "Non posso verificare notizie in tempo reale da questa chat locale. Per domande attuali conviene controllare fonti "
        "aggiornate, comunicati ufficiali, documenti finanziari e testate affidabili. Se mi incolli un articolo o un comunicato, "
        "posso riassumerlo e aiutarti a capirne i punti chiave."
    )


def simple_math_response(prompt):
    matches = [match.group(1).strip() for match in re.finditer(r"([-+*/(). 0-9]+)", prompt.replace(",", "."))]
    expressions = [expr for expr in matches if re.search(r"\d", expr)]
    if not expressions:
        return None
    expr = max(expressions, key=len)
    if not expr or not re.search(r"\d", expr):
        return None
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](eval_node(node.operand))
        raise ValueError("espressione non supportata")

    try:
        result = eval_node(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"Il risultato è {result}."


def writing_response(prompt):
    p = prompt.lower()
    if "email" in p or "mail" in p:
        return (
            "Certo. Ecco una bozza professionale:\n\n"
            "Oggetto: Richiesta informazioni\n\n"
            "Buongiorno,\n"
            "la contatto per chiederle maggiori informazioni in merito alla richiesta indicata. "
            "Resto a disposizione per eventuali dettagli e la ringrazio in anticipo per il riscontro.\n\n"
            "Cordiali saluti"
        )
    if "lista" in p or "checklist" in p:
        return "Ecco una lista ordinata: 1. chiarire l'obiettivo; 2. raccogliere i dati; 3. fare una prima versione; 4. verificare; 5. migliorare il risultato."
    if "piano" in p or "roadmap" in p:
        return "Ti propongo un piano semplice: definire l'obiettivo, dividere il lavoro in passi piccoli, verificare ogni passo, poi rifinire e documentare il risultato."
    return None


def professional_response(prompt, history=None):
    p = prompt.lower().strip()
    if any(greet in p for greet in ["ciao", "salve", "buongiorno", "buonasera", "hey"]):
        if any(term in p for term in ["come stai", "tutto bene"]):
            return "Ciao! Sto funzionando correttamente in locale e sono pronto ad aiutarti sul progetto MiniLLM."
        if any(name in p for name in ["chiami", "nome", "sei"]):
            return "Ciao! Mi chiamo MiniLLM, sono un assistente locale basato sul progetto MiniLLM-32M."
        return "Ciao! Sono MiniLLM. Dimmi pure cosa vuoi fare o che domanda hai."
    if any(phrase in p for phrase in ["cosa puoi fare", "che puoi fare", "a cosa servi"]):
        return (
            "Posso aiutarti a usare MiniLLM in locale: chat sui checkpoint, spiegazioni tecniche, riassunti, piccole bozze, "
            "controllo del training e lettura dei benchmark. Quando una cosa richiede dati aggiornati, te lo segnalo."
        )
    math_answer = simple_math_response(prompt) if any(term in p for term in ["quanto fa", "calcola", "+", "-", "*", "/"]) else None
    if math_answer:
        return math_answer
    if any(term in p for term in ["news", "nws", "notizie", "oggi", "ipo", "borsa", "azioni", "spacex"]):
        return current_info_response()
    if "self-attention" in p or "self attention" in p or "self attention" in p or "attenzione" in p:
        return self_attention_response()
    if any(term in p for term in ["raccontami una storia", "scrivi una storia", "storia di"]):
        return story_response(prompt)
    write_answer = writing_response(prompt)
    if write_answer:
        return write_answer
    if "riassumi" in p or "riassunto" in p:
        topic = extract_topic(prompt, [r"riassumi(?:mi)?\s+(.+)", r"riassunto\s+di\s+(.+)"])
        if topic:
            return f"Posso farlo. Incollami il testo completo su {topic} e ti preparo un riassunto chiaro in punti essenziali."
        return "Certo. Incollami il testo da riassumere e ti restituisco una versione breve, ordinata e facile da leggere."
    if p.startswith("spiegami") or p.startswith("spiega"):
        topic = extract_topic(prompt, [r"spiegami(?:\s+in modo semplice)?\s+(.+)", r"spiega(?:\s+in modo semplice)?\s+(.+)"])
        if topic:
            return (
                f"In breve: {sentence_case(topic)} è un concetto che conviene capire partendo dall'idea principale, "
                "poi dagli esempi e infine dai dettagli. Posso anche scomporlo in passaggi semplici se vuoi."
            )
        return "Certo. Dimmi quale concetto vuoi capire e te lo spiego in modo semplice, con un esempio pratico."
    if p.endswith("?"):
        return (
            "Posso aiutarti, ma con questo checkpoint locale preferisco rispondere in modo prudente. "
            "Dammi un po' di contesto o chiedimi una spiegazione, un riassunto o un esempio concreto."
        )
    return None


def canned_response(prompt):
    return professional_response(prompt)


def looks_incoherent(text):
    cleaned = clean_response(text)
    tokens = words(cleaned)
    if len(cleaned) < 2:
        return True
    if len(tokens) < 3 and len(cleaned) < 24:
        return True
    suspicious = ["text-align", "GDP_", "==", "|||", "@@", "\\", "{", "}", "_", "nbsp", "href"]
    if any(item.lower() in cleaned.lower() for item in suspicious):
        return True
    alpha_ratio = sum(ch.isalpha() or ch.isspace() or ch in ".,;:!?'-" for ch in cleaned) / max(1, len(cleaned))
    if alpha_ratio < 0.72:
        return True
    if tokens:
        unique_ratio = len(set(tokens)) / len(tokens)
        if len(tokens) > 12 and unique_ratio < 0.45:
            return True
        known_ratio = sum(1 for token in tokens if token in COMMON_WORDS or len(token) <= 3) / max(1, len(tokens))
        if len(tokens) > 8 and known_ratio < 0.28:
            return True
        short_fragment_ratio = sum(1 for token in tokens if len(token) <= 2) / max(1, len(tokens))
        if len(tokens) > 12 and short_fragment_ratio > 0.35:
            return True
    glued = sum(1 for token in tokens if len(token) > 18)
    if glued >= 2:
        return True
    return False


def fallback_response(prompt, candidate):
    canned = canned_response(prompt)
    if canned:
        return canned
    if looks_incoherent(candidate):
        return (
            "Non ho generato una risposta affidabile con questo checkpoint. "
            "Il modello MiniLLM-32M è ancora piccolo e non completamente instruction-tuned; "
            "posso comunque aiutarti con domande semplici o puoi provare un checkpoint fine-tuned."
        )
    return clean_response(candidate)


def chunk_text(text, size=10):
    for start in range(0, len(text), size):
        yield text[start:start + size]


def make_app(runtime):
    app = FastAPI(title="MiniLLM Chat Mode")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return CHAT_HTML

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page():
        return CHAT_HTML

    @app.get("/api/chat/checkpoints")
    def checkpoints():
        return {"active_checkpoint": runtime.active_checkpoint, "checkpoints": runtime.list_checkpoints()}

    @app.post("/api/chat")
    async def chat(payload: dict):
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return JSONResponse({"error": "prompt vuoto"}, status_code=400)
        history = normalize_history(payload.get("history", []))
        checkpoint_path = payload.get("checkpoint_path") or payload.get("checkpoint") or runtime.active_checkpoint
        temperature = min(max(float(payload.get("temperature", 0.45)), 0.1), 1.2)
        top_p = min(max(float(payload.get("top_p", 0.82)), 0.1), 1.0)
        top_k = int(payload.get("top_k", 35))
        max_tokens = min(max(int(payload.get("max_tokens", payload.get("max_new_tokens", 80))), 1), 160)
        stream = bool(payload.get("stream", False))
        direct_response = professional_response(prompt, history)
        if direct_response:
            new_history = [*history, {"role": "user", "content": prompt}, {"role": "assistant", "content": direct_response}]
            if stream:
                async def direct_events():
                    for piece in chunk_text(direct_response):
                        yield f"data: {json.dumps({'token': piece})}\n\n"
                        await asyncio.sleep(0)
                    yield f"data: {json.dumps({'done': True, 'response': direct_response, 'history': new_history})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(direct_events(), media_type="text/event-stream")
            return {"response": direct_response, "history": new_history, "checkpoint": runtime.active_checkpoint, "source": "professional_layer"}

        prompt_text = build_prompt(prompt, history)
        model, device = runtime.load(checkpoint_path)

        generated_parts = []
        candidate = ""
        for token, cleaned in generate_tokens(model, runtime.tokenizer, device, prompt_text, max_tokens, temperature, top_p, top_k):
            generated_parts.append(token)
            candidate = cleaned
        response = fallback_response(prompt, candidate or "".join(generated_parts))

        if stream:
            async def events():
                for piece in chunk_text(response):
                    yield f"data: {json.dumps({'token': piece})}\n\n"
                    await asyncio.sleep(0)
                new_history = [*history, {"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                yield f"data: {json.dumps({'done': True, 'response': response, 'history': new_history})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        new_history = [*history, {"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        return {"response": response, "history": new_history, "checkpoint": runtime.active_checkpoint}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Avvia MiniLLM Chat Mode.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best.pt")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--checkpoint_dir", default="models/checkpoints")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


DEFAULT_ARGS = parse_args() if __name__ == "__main__" else argparse.Namespace(
    checkpoint="models/checkpoints/best.pt",
    tokenizer="tokenizer/tokenizer.json",
    checkpoint_dir="models/checkpoints",
    host="127.0.0.1",
    port=8020,
    device="auto",
    no_fp16=False,
)
runtime = ChatRuntime(
    checkpoint=DEFAULT_ARGS.checkpoint,
    tokenizer_path=DEFAULT_ARGS.tokenizer,
    checkpoint_dir=DEFAULT_ARGS.checkpoint_dir,
    device_name=DEFAULT_ARGS.device,
    fp16=not DEFAULT_ARGS.no_fp16,
)
app = make_app(runtime)


def main():
    import uvicorn

    print(f"MiniLLM Chat Mode: http://{DEFAULT_ARGS.host}:{DEFAULT_ARGS.port}/chat")
    print(f"checkpoint: {DEFAULT_ARGS.checkpoint}")
    uvicorn.run(app, host=DEFAULT_ARGS.host, port=DEFAULT_ARGS.port)


if __name__ == "__main__":
    main()
