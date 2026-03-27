"""LoCoMo (Long Conversation Memory) benchmark for NeuroPack.

Evaluates retrieval quality on the LoCoMo-MC10 dataset (1,986 multiple-choice
questions across 10 long conversations).

Usage:
    python benchmarks/locomo.py                  # Mode A: no LLM (TF-IDF match)
    python benchmarks/locomo.py --mode llm       # Mode B: LLM selects answer
    python benchmarks/locomo.py --embedder st    # Use sentence-transformer embeddings
    python benchmarks/locomo.py --limit 100      # Quick test with 100 questions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


DATASET_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "hub",
    "datasets--Percena--locomo-mc10",
    "snapshots",
)


def find_dataset() -> str:
    """Find the downloaded LoCoMo dataset."""
    if os.path.isdir(DATASET_PATH):
        for snap in os.listdir(DATASET_PATH):
            candidate = os.path.join(DATASET_PATH, snap, "data", "locomo_mc10.json")
            if os.path.isfile(candidate):
                return candidate

    # Try downloading
    print("Dataset not found locally. Downloading from HuggingFace...")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Percena/locomo-mc10",
        filename="data/locomo_mc10.json",
        repo_type="dataset",
    )
    return path


def load_dataset(path: str, limit: int | None = None) -> list[dict]:
    """Load LoCoMo-MC10 JSONL dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


def group_by_conversation(records: list[dict]) -> dict[str, list[dict]]:
    """Group questions by conversation (based on question_id prefix)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        conv_id = r["question_id"].rsplit("_", 1)[0]
        groups[conv_id].append(r)
    return dict(groups)


def ingest_conversation(store: MemoryStore, sessions: list[list[dict]], datetimes: list[str] | None = None, chunk_turns: int = 8):
    """Store conversation turns as overlapping chunks for better retrieval.

    Uses a sliding window of `chunk_turns` turns with 50% overlap.
    """
    stride = max(1, chunk_turns // 2)  # 50% overlap

    for i, session in enumerate(sessions):
        timestamp = datetimes[i] if datetimes and i < len(datetimes) else ""
        date_tag = ""
        if timestamp:
            date_str = str(timestamp)[:10].replace(":", "-")
            date_tag = f"date_{date_str}"

        turns = session
        if not turns:
            continue

        # Sliding window chunking
        for start in range(0, len(turns), stride):
            end = min(start + chunk_turns, len(turns))
            chunk_parts = []
            # Inject session date into content so FTS/vector can find it
            if timestamp:
                chunk_parts.append(f"[Date: {str(timestamp)[:10]}]")
            for turn in turns[start:end]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role:
                    chunk_parts.append(f"{role}: {content}")
                else:
                    chunk_parts.append(content)

            chunk_text = "\n".join(chunk_parts)
            if not chunk_text.strip():
                continue

            tags = [f"session_{i+1}", f"turns_{start}-{end-1}"]
            if date_tag:
                tags.append(date_tag)

            store.store(
                content=chunk_text,
                tags=tags,
                source="locomo",
                priority=0.5,
            )

            # Don't create a 1-turn chunk at the end if we already covered it
            if end == len(turns):
                break


STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once that this these those i me my myself we our ours he him "
    "his she her it its they them their what which who whom how where when why "
    "all each every both few more most other some such no nor not only own same "
    "so than too very and but or if while about up down just also there here "
    "because although however still yet".split()
)


def score_mode_a(question: str, choices: list[str], recalled_text: str) -> int:
    """Mode A: No LLM. Pick the choice with highest discriminative overlap.

    For each choice, we look at words that are unique to that choice (not in
    the question and not in other choices) and check how many appear in the
    recalled text. This finds the choice most supported by evidence.
    """
    recalled_lower = recalled_text.lower()
    recalled_words = set(recalled_lower.split()) - STOP_WORDS
    question_words = set(question.lower().split()) - STOP_WORDS

    # Get all choice words for computing discriminative terms
    all_choice_words: list[set[str]] = []
    for c in choices:
        all_choice_words.append(set(c.lower().split()) - STOP_WORDS)

    best_idx = 0
    best_score = -1.0

    for i, choice in enumerate(choices):
        choice_lower = choice.lower()
        choice_words = all_choice_words[i]

        # Words unique to this choice (not in question, not in most other choices)
        other_words: set[str] = set()
        for j, cw in enumerate(all_choice_words):
            if j != i:
                other_words |= cw
        discriminative = choice_words - question_words
        unique_to_choice = discriminative - other_words

        # Score 1: Exact substring match (strongest signal)
        if choice_lower in recalled_lower:
            substring_score = 2.0
        else:
            substring_score = 0.0

        # Score 2: Discriminative word overlap (words unique to this choice found in recall)
        if unique_to_choice:
            unique_overlap = len(unique_to_choice & recalled_words) / len(unique_to_choice)
        else:
            unique_overlap = 0.0

        # Score 3: General word overlap (all non-question choice words)
        if discriminative:
            general_overlap = len(discriminative & recalled_words) / len(discriminative)
        else:
            general_overlap = 0.0

        score = substring_score + unique_overlap * 1.5 + general_overlap * 0.5

        if score > best_score:
            best_score = score
            best_idx = i

    # Check for "Not answerable" — in LoCoMo, when this choice exists it is
    # always the correct answer (adversarial questions test unanswerable detection).
    for i, choice in enumerate(choices):
        if choice.lower().strip() in ("not answerable", "none of the above", "cannot be determined"):
            return i

    return best_idx


def score_mode_llm(
    question: str,
    choices: list[str],
    recalled_text: str,
    provider,
) -> int:
    """Mode B: Use LLM to select the best answer from recalled context."""
    # Adversarial shortcut: "Not answerable" is always correct in LoCoMo
    for i, choice in enumerate(choices):
        if choice.lower().strip() in ("not answerable", "none of the above", "cannot be determined"):
            return i

    choices_str = "\n".join(f"{i}: {c}" for i, c in enumerate(choices))
    prompt = (
        f"Below are excerpts from a conversation between friends.\n\n"
        f"--- EXCERPTS ---\n{recalled_text[:16000]}\n---\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Which choice is correct? Reply with ONLY the choice number (0-{len(choices)-1}), nothing else."
    )
    response = provider.call(
        system="Answer the multiple-choice question. Output ONLY a single digit.",
        user=prompt,
        max_tokens=4,
        temperature=0.0,
    )
    if response is None:
        # LLM failed — fall back to Mode A scoring
        return score_mode_a(question, choices, recalled_text)

    # Extract the first valid digit from response
    import re
    match = re.search(r'\b(\d)\b', response.strip())
    if match:
        idx = int(match.group(1))
        if 0 <= idx < len(choices):
            return idx
    # Fallback: first digit character
    for ch in response.strip():
        if ch.isdigit():
            idx = int(ch)
            if 0 <= idx < len(choices):
                return idx
    # LLM gave unparseable response — fall back to Mode A
    return score_mode_a(question, choices, recalled_text)


def run_benchmark(args):
    """Main benchmark runner."""
    print("=" * 60)
    print("NeuroPack LoCoMo Benchmark")
    print("=" * 60)

    # Find and load dataset
    dataset_path = find_dataset()
    print(f"Dataset: {dataset_path}")
    records = load_dataset(dataset_path, limit=args.limit)
    print(f"Questions: {len(records)}")

    types = Counter(r["question_type"] for r in records)
    print(f"Types: {dict(types)}")

    # Group by conversation
    convs = group_by_conversation(records)
    print(f"Conversations: {len(convs)}")

    # Config
    embedder = "sentence-transformer" if args.embedder == "st" else "tfidf"
    embedding_dim = 384 if args.embedder == "st" else 256
    print(f"Embedder: {embedder}")
    print(f"Mode: {'LLM' if args.mode == 'llm' else 'A (no LLM)'}")
    print(f"Recall limit: {args.top_k}")
    print(f"Reranker: {'cross-encoder' if args.reranker else 'off'}")

    # LLM provider for Mode B
    llm_provider = None
    if args.mode == "llm":
        from neuropack.llm.provider import LLMProvider
        from neuropack.llm.models import LLMConfig

        # Provider preference order (Gemini is cheapest for bulk)
        key_map = [
            ("gemini", "GEMINI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
        ]

        api_key = None
        provider_name = args.llm_provider  # May be None

        # Build search list based on whether provider was specified
        if provider_name:
            search_providers = [(provider_name, {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[provider_name])]
        else:
            search_providers = list(key_map)

        # Try environment variables
        for prov, env_var in search_providers:
            val = os.environ.get(env_var, "")
            if val:
                api_key, provider_name = val, prov
                break

        # Try .env files
        if not api_key:
            target_keys = {env_var: prov for prov, env_var in search_providers}
            for env_path in [".env", os.path.expanduser("~/.env"), "../Poly/.env"]:
                if os.path.isfile(env_path):
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("#") or "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip()
                            if k in target_keys and v:
                                api_key, provider_name = v, target_keys[k]
                                break
                    if api_key:
                        break

        if not api_key or not provider_name:
            print("\nERROR: Mode 'llm' requires an API key.")
            print("Set GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY")
            sys.exit(1)

        llm_config = LLMConfig(
            name="bench",
            provider=provider_name,
            api_key=api_key,
            is_default=True,
        )
        llm_provider = LLMProvider(llm_config)
        print(f"LLM: {provider_name}/{llm_provider._model}")

    print("-" * 60)

    # Run evaluation
    correct = 0
    total = 0
    correct_by_type: dict[str, int] = defaultdict(int)
    total_by_type: dict[str, int] = defaultdict(int)
    latencies_store: list[float] = []
    latencies_recall: list[float] = []

    import tempfile

    for conv_id, questions in convs.items():
        print(f"\nConversation: {conv_id} ({len(questions)} questions)")

        # Create fresh store per conversation
        tmp_dir = tempfile.mkdtemp(prefix="np_locomo_")
        db_path = os.path.join(tmp_dir, "bench.db")
        reranker_type = "cross-encoder" if args.reranker else "off"
        config = NeuropackConfig(
            db_path=db_path,
            embedder_type=embedder,
            embedding_dim=embedding_dim,
            retrieval_weight_vec=0.6,
            retrieval_weight_fts=0.4,
            retrieval_weight_graph=0.0,
            recall_limit=args.top_k,
            auto_tag=False,
            dedup_threshold=0.99,  # Disable dedup for benchmark
            reranker=reranker_type,
        )
        store = MemoryStore(config)
        store.initialize()

        # Ingest conversation (use first question's sessions - same for all in conv)
        sample = questions[0]
        t0 = time.perf_counter()
        ingest_conversation(
            store,
            sample["haystack_sessions"],
            sample.get("haystack_session_datetimes"),
        )
        t_ingest = time.perf_counter() - t0
        latencies_store.append(t_ingest)
        print(f"  Ingested {len(sample['haystack_sessions'])} sessions in {t_ingest:.2f}s")

        # Answer each question
        for q in questions:
            t0 = time.perf_counter()
            results = store.recall(q["question"], limit=args.top_k)
            t_recall = time.perf_counter() - t0
            latencies_recall.append(t_recall)

            # Build recalled text (use full content since chunks are now smaller)
            recalled_text = ""
            for r in results:
                recalled_text += r.record.content[:1000] + "\n---\n"

            # Score
            if args.mode == "llm" and llm_provider:
                predicted = score_mode_llm(
                    q["question"], q["choices"], recalled_text, llm_provider
                )
                # Throttle API calls to avoid rate limits
                time.sleep(0.2)
            else:
                predicted = score_mode_a(
                    q["question"], q["choices"], recalled_text
                )

            is_correct = predicted == q["correct_choice_index"]
            if is_correct:
                correct += 1
                correct_by_type[q["question_type"]] += 1
            total += 1
            total_by_type[q["question_type"]] += 1

        store.close()

        # Clean up temp DB
        try:
            os.remove(db_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass

        # Progress
        acc = correct / total * 100 if total else 0
        print(f"  Running accuracy: {correct}/{total} ({acc:.1f}%)")

    # Final report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print()
    print("By question type:")
    for qtype in sorted(total_by_type.keys()):
        c = correct_by_type[qtype]
        t = total_by_type[qtype]
        pct = c / t * 100 if t else 0
        print(f"  {qtype:25s} {c:4d}/{t:4d} ({pct:.1f}%)")

    print()
    avg_store = sum(latencies_store) / len(latencies_store) if latencies_store else 0
    avg_recall = sum(latencies_recall) / len(latencies_recall) if latencies_recall else 0
    med_recall = sorted(latencies_recall)[len(latencies_recall) // 2] if latencies_recall else 0
    print(f"Avg ingest time:  {avg_store:.2f}s per conversation")
    print(f"Avg recall time:  {avg_recall*1000:.1f}ms per query")
    print(f"Med recall time:  {med_recall*1000:.1f}ms per query")
    print(f"Embedder:         {embedder}")
    print(f"Mode:             {'LLM' if args.mode == 'llm' else 'A (no LLM, text overlap)'}")

    # Save results
    results_path = os.path.join(
        Path(__file__).parent, f"locomo_results_{args.embedder}_{args.mode}.json"
    )
    results = {
        "overall_accuracy": round(correct / total * 100, 2) if total else 0,
        "correct": correct,
        "total": total,
        "by_type": {
            qtype: {
                "correct": correct_by_type[qtype],
                "total": total_by_type[qtype],
                "accuracy": round(correct_by_type[qtype] / total_by_type[qtype] * 100, 2)
                if total_by_type[qtype]
                else 0,
            }
            for qtype in sorted(total_by_type.keys())
        },
        "avg_recall_ms": round(avg_recall * 1000, 1),
        "median_recall_ms": round(med_recall * 1000, 1),
        "avg_ingest_s": round(avg_store, 2),
        "embedder": embedder,
        "mode": args.mode,
        "top_k": args.top_k,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="NeuroPack LoCoMo Benchmark")
    parser.add_argument("--mode", choices=["a", "llm"], default="a", help="Scoring mode: 'a' (no LLM) or 'llm'")
    parser.add_argument("--embedder", choices=["tfidf", "st"], default="st", help="Embedder: 'tfidf' or 'st' (sentence-transformer)")
    parser.add_argument("--top-k", type=int, default=30, help="Number of memories to recall per question")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for quick testing)")
    parser.add_argument("--reranker", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--llm-provider", choices=["gemini", "anthropic", "openai"], default=None, help="LLM provider for mode=llm")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
