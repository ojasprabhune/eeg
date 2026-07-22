"""
Generate AAC-style training corpora for the language model, one per experiment.

Each experiment can only decode a small set of letters (see gesture_experiments
in eeg/gesture2hand/utils/gestures.py), so every sentence in that experiment's
corpus must be spelled with only those letters *and* be made of real English
words. Filtering the LM to allowed *letters* is not enough: it emits letter-soup
like "oaterain" that uses the right letters but is not English.

Instead we constrain generation with a trie of a curated vocabulary (all real
words spelled with the experiment's letters). At every step the LM may only
pick a token that keeps the current word on a path to a real word, so every
word that comes out is valid; the LM still chooses natural word orderings.

Decoding is trie-constrained multinomial sampling (not beam search: beam search
collapses to one high-probability sentence, killing the diversity a corpus
needs). We sample many newline-separated sentences per pass, reusing the KV
cache, and anneal the temperature upward when a pass stops finding new
sentences so we can fill the target even for tiny letter sets.

Run:  python scripts/language_model/corpus_gen.py
Writes eeg/language_model/data/<experiment>_corpus.txt for each experiment.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The eeg package __init__ pulls in hardware/vision deps (pywinusb, mediapipe)
# that a minimal CUDA container doesn't have. gesture_experiments and Trie are
# dependency-free, so fall back to vendored copies (copied into the image).
try:
    from eeg.gesture2hand.utils.gestures import gesture_experiments
    from eeg.trie import Trie
except Exception:  # pragma: no cover - container path
    from gestures import gesture_experiments
    from trie import Trie

# --- configuration -----------------------------------------------------------

MODEL_NAME = os.environ.get("CORPUS_MODEL", "Qwen/Qwen2.5-3B-Instruct")
OUTPUT_DIR = os.environ.get("CORPUS_OUTPUT_DIR", "eeg/language_model/data")


def pick_device() -> str:
    """cuda on the A100 job, mps on the Mac, cpu as a last resort."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = os.environ.get("CORPUS_DEVICE") or pick_device()

BOUNDARY_CHARS = {" ", "\n"}

EXPERIMENTS = ["asl_8_letters", "common_8_letters", "6_letters"]

# Curated vocabularies. Words are filtered at runtime to the experiment's letter
# set, so a word only appears in an experiment whose letters can spell it. Keep
# them common where possible; a few rarer words widen the tiny sets enough to
# reach 500 distinct sentences.
VOCAB = {
    "asl_8_letters": [  # letters: c e f h i o r t
        "he", "hi", "if", "it", "oh", "or", "of", "to", "the", "her", "hit",
        "hot", "fit", "fir", "for", "foe", "hoe", "ice", "toe", "tie", "tic",
        "rot", "cot", "ref", "tee", "fee", "ire", "ore", "roe", "rice", "rich",
        "rite", "tire", "tier", "fire", "hire", "here", "hero", "core", "chef",
        "fore", "fort", "heir", "cite", "fret", "thee", "itch", "etch", "echo",
        "coif", "roof", "reef", "free", "tree", "chic", "choir", "forth",
        "froth", "forte", "there", "three", "chief", "chore", "force", "fetch",
        "other", "ether", "torch", "their", "office", "coffee", "either",
        "effort", "recite", "heifer", "toffee", "rioter", "critic", "richer",
        "hotter", "fitter", "orifice",
    ],
    "common_8_letters": [  # letters: a e i n o r s t
        "a", "an", "i", "in", "is", "it", "its", "on", "at", "as", "no", "not",
        "nor", "or", "so", "to", "one", "ten", "none", "into", "onto", "are",
        "see", "sees", "eat", "eats", "ate", "sit", "sits", "sat", "ran",
        "rain", "rains", "rest", "rests", "rise", "rises", "roast", "stare",
        "stares", "start", "starts", "stir", "taste", "tastes", "snore", "soar",
        "arise", "insist", "resist", "retain", "tree", "trees", "tea", "teas",
        "stone", "stones", "seat", "seats", "nest", "nests", "star", "stars",
        "store", "stores", "train", "trains", "note", "notes", "rose", "roses",
        "tie", "ties", "toe", "toes", "tone", "tones", "tear", "tears", "ear",
        "ears", "iron", "irons", "art", "arts", "oat", "oats", "rat", "rats",
        "ant", "ants", "nose", "sea", "seas", "air", "toast", "nation",
        "nations", "station", "artist", "artists", "senior", "sister",
        "sisters", "street", "streets", "reason", "reasons", "season", "toaster",
        "senate", "near", "neat", "sane", "sore", "torn", "east", "inner",
    ],
    "6_letters": [  # letters: a e i n o t
        "a", "i", "an", "at", "in", "it", "on", "no", "to", "eat", "ate", "tea",
        "toe", "ten", "tan", "tin", "ton", "ant", "oat", "one", "net", "not",
        "tie", "ion", "note", "tone", "neat", "nine", "none", "into", "onto",
        "teen", "tint", "anti", "iota", "atone", "eaten", "titan", "anion",
        "onion", "nation", "intent", "attain", "tenant", "tannin", "innate",
        "anoint", "notation", "annotate", "intention", "attention", "nineteen",
    ],
}

# --- generation hyperparameters ---
NUM_SENTENCES = 500
MIN_WORDS = 3
MAX_WORDS = 8
BATCH = int(os.environ.get("CORPUS_BATCH", 32))  # sampling streams per forward
TOKENS_PER_PASS = int(os.environ.get("CORPUS_TOKENS_PER_PASS", 350))
BASE_TEMPERATURE = 1.0
MAX_TEMPERATURE = 1.9
MAX_PASSES = int(os.environ.get("CORPUS_MAX_PASSES", 25))
REP_PENALTY = float(os.environ.get("CORPUS_REP_PENALTY", 4.0))  # per-repeat logit penalty


# --- trie + token pool -------------------------------------------------------


def build_trie(words: list[str]) -> Trie:
    trie = Trie()
    for word in words:
        trie.insert(word)
    return trie


def enumerate_nodes(trie: Trie) -> list:
    """Return every node in the trie via depth-first traversal."""
    nodes = []
    stack = [trie.root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children.values())
    return nodes


def build_pool(tokenizer, allowed_chars: set) -> list[tuple[int, str]]:
    """All tokens whose text uses only the allowed letters / word boundaries."""
    pool = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if text and all(c in allowed_chars for c in text):
            pool.append((token_id, text))
    return pool


def simulate(text: str, start, root, allowed_letters: set):
    """
    Walk `text` through the trie from `start`. Letters must follow an existing
    trie edge; a space/newline may only appear after a complete word (no empty
    words, no gluing two words), then we jump back to the root. Returns the
    ending node, or None if the token is illegal here.
    """
    node = start
    for c in text:
        if c in allowed_letters:
            nxt = node.children.get(c)
            if nxt is None:
                return None
            node = nxt
        else:  # space or newline (pool only holds allowed chars)
            if not node.is_word:
                return None
            node = root
    return node


def build_transitions(trie: Trie, pool: list[tuple[int, str]], allowed_letters: set):
    """
    Precompute, per trie node, the legal-token id sets (as CPU tensors) for each
    word-count regime, plus the node each token lands on. Masking during
    generation is then a dict lookup + CPU gather, with no per-step tensor
    building. Token roles:

      - letter tokens  : stay inside / complete the current word (no boundary)
      - space tokens   : contain a space -> start the next word (same sentence)
      - newline tokens : contain a newline -> end the current sentence

    Variants (which roles are legal given current word count `wc`):
      - "min"  wc < MIN_WORDS       : letters + spaces (too short to end)
      - "mid"  MIN <= wc < MAX      : letters + spaces + newlines
      - "max"  wc >= MAX_WORDS      : letters + newlines (too long for new word)
    """
    root = trie.root
    node_variants: dict[int, dict[str, torch.Tensor]] = {}
    node_next: dict[int, dict[int, object]] = {}

    for node in enumerate_nodes(trie):
        letters, spaces, newlines, nxt = [], [], [], {}
        for token_id, text in pool:
            end = simulate(text, node, root, allowed_letters)
            if end is None:
                continue
            nxt[token_id] = end
            if "\n" in text:
                newlines.append(token_id)
            elif " " in text:
                spaces.append(token_id)
            else:
                letters.append(token_id)
        node_variants[id(node)] = {
            "min": torch.tensor(letters + spaces, dtype=torch.long),
            "mid": torch.tensor(letters + spaces + newlines, dtype=torch.long),
            "max": torch.tensor(letters + newlines, dtype=torch.long),
        }
        node_next[id(node)] = nxt

    return node_variants, node_next


# --- model / prompt ----------------------------------------------------------


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # bf16 on GPU; float32 on CPU (CPU bf16 matmul is emulated and slow)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=dtype, low_cpu_mem_usage=True
        )
    except TypeError:  # older transformers uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer


def build_prompt_ids(tokenizer, model, allowed_letters: set, words: list[str]):
    """Steer the LM toward simple AAC sentences within the letter/word set."""
    letters = ", ".join(sorted(allowed_letters))
    sample_words = ", ".join(words[: min(len(words), 40)])
    system = (
        "You write AAC (augmentative and alternative communication) sentences "
        "for speech-decoding research: short, simple, everyday things a person "
        f"would say. Every word may use ONLY these letters: {letters}. Every "
        "word must be a real English word. Output lowercase, no punctuation, "
        "one sentence per line, nothing else."
    )
    user = (
        f"Write many short sentences (3 to 8 words each) using only the letters "
        f"{letters} and only real English words such as: {sample_words}. Keep "
        "them simple and conversational, one per line."
    )
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer([text], return_tensors="pt").input_ids.to(model.device)


# --- decoding ----------------------------------------------------------------


def generate_pass(model, prompt_ids, root, transitions, tok_text, temperature):
    """
    One trie-constrained sampling pass over BATCH independent streams sharing a
    single batched forward (and KV cache) per step. Returns all sentences the
    streams produced. Batching gives ~BATCH x throughput and more diversity.
    """
    node_variants, node_next = transitions
    device = model.device

    input_ids = prompt_ids.repeat(BATCH, 1)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float().cpu()  # (BATCH, vocab), sample on CPU

    nodes = [root] * BATCH
    buffers = [""] * BATCH
    used = [{} for _ in range(BATCH)]  # content-token counts in current sentence
    sentences: list[str] = []

    for _ in range(TOKENS_PER_PASS):
        next_tokens = []
        for b in range(BATCH):
            node = nodes[b]
            word_count = len(buffers[b].split())
            if word_count < MIN_WORDS:
                allowed = node_variants[id(node)]["min"]
            elif word_count >= MAX_WORDS:
                allowed = node_variants[id(node)]["max"]
            else:
                allowed = node_variants[id(node)]["mid"]

            # gather + sample entirely on CPU (avoids per-op MPS sync stalls);
            # sanitize non-finite logits so multinomial always gets valid probs.
            row = logits[b]
            # repetition penalty: discourage reusing content tokens (words)
            # already emitted in the current sentence, killing "he he he" runs
            if REP_PENALTY and used[b]:
                row = row.clone()
                idx = torch.tensor(list(used[b].keys()))
                cnts = torch.tensor([used[b][k] for k in used[b]], dtype=row.dtype)
                row[idx] -= REP_PENALTY * cnts
            selected = (row.index_select(0, allowed) / temperature).nan_to_num(
                nan=-1e9, posinf=-1e9, neginf=-1e9
            )
            probs = torch.softmax(selected, dim=-1)
            token_id = int(allowed[torch.multinomial(probs, 1).item()])
            next_tokens.append(token_id)

            piece = tok_text[token_id]
            nodes[b] = node_next[id(node)][token_id]
            if "\n" in piece:
                if buffers[b].strip():
                    sentences.append(buffers[b].strip())
                buffers[b] = ""
                nodes[b] = root
                used[b] = {}
            else:
                buffers[b] += piece
                if any(c not in " \n" for c in piece):  # content token
                    used[b][token_id] = used[b].get(token_id, 0) + 1

        step_ids = torch.tensor(next_tokens, device=device).unsqueeze(1)  # (BATCH, 1)
        with torch.no_grad():
            out = model(input_ids=step_ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float().cpu()

    for b in range(BATCH):
        if buffers[b].strip():
            sentences.append(buffers[b].strip())
    return sentences


# --- corpus ------------------------------------------------------------------


def normalize(sentence: str) -> str:
    return " ".join(sentence.lower().split())


def is_valid(sentence: str, vocab: set) -> bool:
    words = sentence.split()
    return MIN_WORDS <= len(words) <= MAX_WORDS and all(w in vocab for w in words)


def generate_corpus(model, tokenizer, experiment: str) -> list[str]:
    allowed_letters = set(gesture_experiments[experiment].keys())
    allowed_chars = allowed_letters | BOUNDARY_CHARS

    # keep only words that fit the experiment's letters (defensive)
    words = [w for w in VOCAB[experiment] if set(w) <= allowed_letters]
    dropped = [w for w in VOCAB[experiment] if set(w) > allowed_letters]
    if dropped:
        print(f"  dropped words not in letter set: {dropped}")
    vocab = set(words)

    trie = build_trie(words)
    pool = build_pool(tokenizer, allowed_chars)
    transitions = build_transitions(trie, pool, allowed_letters)
    tok_text = {tid: text for tid, text in pool}
    prompt_ids = build_prompt_ids(tokenizer, model, allowed_letters, words)

    print(
        f"  letters={sorted(allowed_letters)} words={len(words)} "
        f"pool={len(pool)} nodes={len(transitions[0])}"
    )

    sentences: list[str] = []
    seen: set = set()
    temperature = BASE_TEMPERATURE
    empty_streak = 0

    for _ in range(MAX_PASSES):
        if len(sentences) >= NUM_SENTENCES:
            break
        raw = generate_pass(model, prompt_ids, trie.root, transitions, tok_text, temperature)
        new_count = 0
        for sentence in raw:
            sentence = normalize(sentence)
            if sentence and sentence not in seen and is_valid(sentence, vocab):
                seen.add(sentence)
                sentences.append(sentence)
                new_count += 1
                if len(sentences) >= NUM_SENTENCES:
                    break

        # anneal temperature up when a pass finds little new; stop if fully dry
        if new_count < 5:
            temperature = min(MAX_TEMPERATURE, temperature + 0.15)
        if new_count == 0:
            empty_streak += 1
            if empty_streak >= 3:
                print(f"  exhausted at {len(sentences)} sentences (vocab too small).")
                break
        else:
            empty_streak = 0
        print(f"  {len(sentences):>4}/{NUM_SENTENCES}  (temp={temperature:.2f})")

    return sentences


def main():
    import sys

    experiments = sys.argv[1:] or EXPERIMENTS
    unknown = [e for e in experiments if e not in gesture_experiments]
    if unknown:
        raise SystemExit(f"unknown experiment(s): {unknown}")

    print("Loading model...")
    model, tokenizer = load_model()

    written = []
    for experiment in experiments:
        print(f"\n=== {experiment} ===")
        sentences = generate_corpus(model, tokenizer, experiment)
        path = f"{OUTPUT_DIR}/{experiment}_corpus.txt"
        with open(path, "w") as file:
            file.write("\n".join(sentences) + "\n")
        print(f"Wrote {len(sentences)} sentences to {path}")
        written.append(path)

    # optional: upload corpora to Azure Blob (used by the Container Apps GPU job)
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container = os.environ.get("BLOB_CONTAINER")
    if conn and container:
        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient.from_connection_string(conn)
        try:
            client.create_container(container)
        except Exception:
            pass
        for path in written:
            name = os.path.basename(path)
            with open(path, "rb") as data:
                client.get_blob_client(container, name).upload_blob(data, overwrite=True)
            print(f"Uploaded {name} to blob container '{container}'")


if __name__ == "__main__":
    main()
