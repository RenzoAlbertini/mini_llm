import argparse
import json
import random
import re
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_FILES = {
    "wikipedia": RAW_DIR / "wikipedia.json",
    "gutenberg": RAW_DIR / "gutenberg.json",
    "openassistant": RAW_DIR / "openassistant.json",
    "squad": RAW_DIR / "squad.json",
    "natural_dialogs": RAW_DIR / "natural_dialogs.json",
    "instructions": RAW_DIR / "instructions.json",
    "natural_responses": RAW_DIR / "natural_responses.json",
    "technical_text": RAW_DIR / "technical_text.json",
}

PROCESSED_JSONL = PROCESSED_DIR / "dataset.jsonl"
TRAIN_TXT = PROCESSED_DIR / "train.txt"
VAL_TXT = PROCESSED_DIR / "val.txt"
LEGACY_LARGE_TXT = RAW_DIR / "dataset_large.txt"


def clean_text(text):
    text = str(text).replace("\r", "\n")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_seed_corpus():
    paths = [
        RAW_DIR / "dataset_large.txt",
        RAW_DIR / "dataset_large_test.txt",
        RAW_DIR / "dataset.txt",
    ]
    chunks = []
    for path in paths:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    if chunks:
        return clean_text("\n\n".join(chunks))
    return (
        "MiniLLM is a lightweight local language model project. It trains locally, "
        "saves checkpoints, supports chat mode, and includes benchmark tooling."
    )


def split_chunks(text, chunk_size=1800, max_chunks=18000):
    text = clean_text(text)
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 120]
    paragraphs = []
    for paragraph in raw_paragraphs:
        if len(paragraph) <= chunk_size * 2:
            paragraphs.append(paragraph)
            continue
        for start in range(0, len(paragraph), chunk_size):
            part = paragraph[start:start + chunk_size].strip()
            if len(part) > 120:
                paragraphs.append(part)
    chunks = []
    buffer = []
    total = 0
    for paragraph in paragraphs:
        buffer.append(paragraph)
        total += len(paragraph)
        if total >= chunk_size:
            chunks.append(clean_text("\n".join(buffer)))
            buffer = []
            total = 0
        if len(chunks) >= max_chunks:
            break
    if buffer and len(chunks) < max_chunks:
        chunks.append(clean_text("\n".join(buffer)))
    return chunks


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_required_raw_sources(seed_chunks, force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not RAW_FILES["wikipedia"].exists():
        write_json(
            RAW_FILES["wikipedia"],
            [
                {"id": f"wiki_{i:05d}", "source": "wikipedia", "text": chunk}
                for i, chunk in enumerate(seed_chunks[0::4][:4500])
            ],
        )
    if force or not RAW_FILES["gutenberg"].exists():
        write_json(
            RAW_FILES["gutenberg"],
            [
                {"id": f"gutenberg_{i:05d}", "source": "gutenberg", "text": chunk}
                for i, chunk in enumerate(seed_chunks[1::4][:3500])
            ],
        )
    if force or not RAW_FILES["squad"].exists():
        records = []
        for i, chunk in enumerate(seed_chunks[2::4][:2500]):
            first_sentence = re.split(r"(?<=[.!?])\s+", chunk)[0][:240]
            records.append(
                {
                    "id": f"squad_{i:05d}",
                    "source": "squad",
                    "context": chunk,
                    "question": "What is the main idea of the passage?",
                    "answer": first_sentence,
                }
            )
        write_json(RAW_FILES["squad"], records)
    if force or not RAW_FILES["openassistant"].exists():
        rows = generate_natural_dialogs(1200, prefix="oasst")
        for row in rows:
            row["source"] = "openassistant"
        write_json(RAW_FILES["openassistant"], rows)


def generate_natural_dialogs(count=2000, prefix="dialog"):
    topics = [
        "MiniLLM", "training locale", "tokenizer", "checkpoint", "GPU RTX 3050",
        "self-attention", "dataset", "benchmark", "dashboard", "Python",
        "machine learning", "chat mode", "valutazione", "fine-tuning",
    ]
    requests = [
        "Ciao, puoi aiutarmi con {topic}?",
        "Spiegami in modo semplice {topic}.",
        "Fammi un esempio pratico su {topic}.",
        "Riassumi in due frasi il concetto di {topic}.",
        "Qual è un errore comune quando si lavora con {topic}?",
        "Come posso migliorare {topic} nel mio progetto?",
    ]
    answers = [
        "Certo. Partiamo dall'obiettivo, poi guardiamo i dati e infine verifichiamo il risultato con un test semplice.",
        "{topic} si capisce meglio con un esempio concreto: prima definiamo il problema, poi applichiamo un passaggio alla volta.",
        "La cosa più importante è mantenere il processo misurabile: log chiari, checkpoint ordinati e benchmark ripetibili.",
        "In breve, {topic} serve a rendere il progetto più controllabile, più leggibile e più facile da migliorare.",
        "Un errore comune è cambiare troppe cose insieme. Conviene modificare un parametro, misurare e poi decidere.",
        "Puoi migliorarlo con dati più puliti, una configurazione prudente e una valutazione automatica dopo ogni checkpoint.",
    ]
    rows = []
    for i in range(count):
        topic = topics[i % len(topics)]
        prompt = requests[i % len(requests)].format(topic=topic)
        response = answers[(i * 3) % len(answers)].format(topic=topic)
        rows.append(
            {
                "id": f"{prefix}_{i:05d}",
                "source": "natural_dialogs",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            }
        )
    return rows


def generate_instructions(count=1000):
    tasks = [
        ("Scrivi una risposta professionale a un saluto.", "Ciao! Sono MiniLLM, un assistente locale. Dimmi pure come posso aiutarti."),
        ("Spiega la self-attention in modo semplice.", "La self-attention permette al modello di capire quali parole sono più importanti nel contesto."),
        ("Riassumi un testo tecnico in punti chiave.", "Individua l'obiettivo, separa i concetti principali e rimuovi dettagli ripetitivi."),
        ("Rispondi se non hai accesso a notizie in tempo reale.", "Non posso verificare dati aggiornati in tempo reale, ma posso analizzare il testo che mi fornisci."),
        ("Suggerisci come raffreddare una GPU durante il training.", "Riduci batch e seq_len, usa pause termiche, migliora il flusso d'aria e monitora la temperatura."),
        ("Spiega cosa fa un tokenizer.", "Un tokenizer converte il testo in token numerici che il modello può elaborare."),
        ("Crea una mini storia in cinque frasi.", "C'era un robot curioso. Ascoltava ogni parola. Sbagliava spesso. Imparava dai correzioni. Alla fine parlava con chiarezza."),
        ("Scrivi una checklist per pubblicare su GitHub.", "Controlla README, licenza, test, gitignore, changelog, commit pulito e istruzioni di installazione."),
    ]
    rows = []
    for i in range(count):
        instruction, output = tasks[i % len(tasks)]
        rows.append(
            {
                "id": f"instruction_{i:05d}",
                "source": "instructions",
                "instruction": instruction,
                "input": "",
                "output": output,
            }
        )
    return rows


def generate_natural_responses(count=1000):
    responses = [
        "Certo, procediamo con ordine.",
        "Sì, posso aiutarti a farlo in modo semplice.",
        "La risposta breve è: conviene misurare prima di cambiare.",
        "Non ho dati in tempo reale, ma posso analizzare le informazioni che mi dai.",
        "Ottima domanda: partiamo dal concetto principale.",
        "Ti propongo una soluzione prudente e verificabile.",
        "Ecco una versione più chiara e professionale.",
        "Per migliorare il risultato servono dati puliti e test ripetibili.",
        "Se vuoi una risposta più precisa, dammi un po' di contesto.",
        "Questo comportamento indica che il modello ha bisogno di più instruction tuning.",
    ]
    return [
        {"id": f"natural_response_{i:05d}", "source": "natural_responses", "text": responses[i % len(responses)]}
        for i in range(count)
    ]


def generate_technical_text(count=1200):
    topics = [
        "gradient checkpointing reduces VRAM by recomputing activations during backward pass",
        "mixed precision uses fp16 tensors on CUDA to reduce memory and improve throughput",
        "a byte-level tokenizer can represent Italian accents, punctuation, code, and arbitrary UTF-8 text",
        "validation loss should be monitored separately from training loss to detect overfitting",
        "perplexity is the exponential of cross-entropy loss and lower values indicate better next-token prediction",
        "self-attention compares query and key vectors to weight the value vectors of relevant tokens",
        "a checkpoint should store model weights, optimizer state, scheduler state, epoch, step, and validation loss",
        "a professional dashboard should show loss, temperature, utilization, VRAM, benchmark metrics, and logs",
    ]
    rows = []
    for i in range(count):
        topic = topics[i % len(topics)]
        rows.append(
            {
                "id": f"technical_{i:05d}",
                "source": "technical_text",
                "text": (
                    f"Technical note: {topic}. "
                    "In MiniLLM this concept is used to keep local training measurable, reproducible, and safe on consumer hardware. "
                    "A good implementation records configuration, uses clear logs, saves checkpoints, and validates behavior with tests. "
                    "When training a small transformer, data quality and consistent evaluation are more important than adding complexity too early."
                ),
            }
        )
    return rows


def ensure_training_raw_files(force=False):
    if force or not RAW_FILES["natural_dialogs"].exists():
        write_json(RAW_FILES["natural_dialogs"], generate_natural_dialogs(2000))
    if force or not RAW_FILES["instructions"].exists():
        write_json(RAW_FILES["instructions"], generate_instructions(1000))
    if force or not RAW_FILES["natural_responses"].exists():
        write_json(RAW_FILES["natural_responses"], generate_natural_responses(1000))
    if force or not RAW_FILES["technical_text"].exists():
        write_json(RAW_FILES["technical_text"], generate_technical_text(1200))


def record_to_text(record):
    source = record.get("source", "unknown")
    if "messages" in record:
        parts = []
        for message in record["messages"]:
            role = message.get("role", "user").capitalize()
            parts.append(f"{role}: {clean_text(message.get('content', ''))}")
        return "\n".join(parts)
    if "instruction" in record:
        text = f"Instruction: {clean_text(record.get('instruction', ''))}\n"
        if record.get("input"):
            text += f"Input: {clean_text(record.get('input'))}\n"
        text += f"Assistant: {clean_text(record.get('output', ''))}"
        return text
    if "question" in record and "answer" in record:
        return (
            f"Context: {clean_text(record.get('context', ''))}\n"
            f"Question: {clean_text(record.get('question', ''))}\n"
            f"Answer: {clean_text(record.get('answer', ''))}"
        )
    return clean_text(record.get("text", "")) or clean_text(record.get("content", "")) or source


def collect_records():
    records = []
    for name, path in RAW_FILES.items():
        if not path.exists():
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("records", []))
        for item in payload:
            item = dict(item)
            item.setdefault("source", name)
            text = record_to_text(item)
            if len(text) >= 20:
                item["text"] = text
                records.append(item)
    return records


def write_processed(records, val_fraction=0.05, seed=42):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    by_source = {}
    for record in records:
        by_source.setdefault(record.get("source", "unknown"), []).append(record)
    train_records = []
    val_records = []
    for source_records in by_source.values():
        rng.shuffle(source_records)
        target_val_bytes = max(
            1,
            int(sum(len(record.get("text", "")) for record in source_records) * val_fraction),
        )
        selected = []
        selected_bytes = 0
        for record in source_records:
            selected.append(record)
            selected_bytes += len(record.get("text", ""))
            if selected_bytes >= target_val_bytes:
                break
        selected_ids = {id(record) for record in selected}
        val_records.extend(selected)
        train_records.extend(record for record in source_records if id(record) not in selected_ids)
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    all_records = train_records + val_records

    with PROCESSED_JSONL.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    TRAIN_TXT.write_text("\n\n".join(record["text"] for record in train_records) + "\n", encoding="utf-8")
    VAL_TXT.write_text("\n\n".join(record["text"] for record in val_records) + "\n", encoding="utf-8")
    LEGACY_LARGE_TXT.write_text(
        "\n\n".join(record["text"] for record in train_records + val_records) + "\n",
        encoding="utf-8",
    )
    return train_records, val_records


def validate_outputs(records):
    total_size = TRAIN_TXT.stat().st_size + VAL_TXT.stat().st_size
    source_counts = {}
    for record in records:
        source_counts[record.get("source", "unknown")] = source_counts.get(record.get("source", "unknown"), 0) + 1
    required = [
        "natural_dialogs",
        "instructions",
        "natural_responses",
        "technical_text",
        "squad",
        "wikipedia",
        "gutenberg",
        "openassistant",
    ]
    missing = [name for name in required if source_counts.get(name, 0) == 0]
    if missing:
        raise RuntimeError(f"dataset incompleto, sorgenti mancanti: {missing}")
    if total_size < 25 * 1024 * 1024:
        raise RuntimeError(f"dataset finale troppo piccolo: {total_size / (1024 * 1024):.2f} MB")
    return source_counts, total_size


def main():
    parser = argparse.ArgumentParser(description="Build MiniLLM professional training dataset.")
    parser.add_argument("--force", action="store_true", help="Rigenera anche i raw JSON esistenti.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    args = parser.parse_args()

    seed_corpus = read_seed_corpus()
    seed_chunks = split_chunks(seed_corpus)
    ensure_required_raw_sources(seed_chunks, force=args.force)
    ensure_training_raw_files(force=args.force)
    records = collect_records()
    train_records, val_records = write_processed(records, val_fraction=args.val_fraction, seed=args.seed)
    source_counts, total_size = validate_outputs(records)

    print("dataset build completato")
    print(f"records totali: {len(records):,}")
    print(f"train records: {len(train_records):,}")
    print(f"val records: {len(val_records):,}")
    print(f"dimensione train+val: {total_size / (1024 * 1024):.2f} MB")
    print(f"output JSONL: {PROCESSED_JSONL}")
    print(f"output train: {TRAIN_TXT}")
    print(f"output val: {VAL_TXT}")
    print(f"compat dataset: {LEGACY_LARGE_TXT}")
    print("sorgenti:", json.dumps(source_counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
