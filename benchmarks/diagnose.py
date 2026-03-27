"""Diagnostic script for NeuroPack LoCoMo multi_hop + temporal_reasoning failures.

Determines whether poor accuracy is caused by:
  (A) RETRIEVAL failure - relevant content not found in top-20, or
  (B) SCORING failure - content is found but Mode-A text matching fails.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# ------------------------------------------------------------------
# Path setup (same as locomo.py)
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore

# Reuse the ingestion helper from the benchmark
from locomo import ingest_conversation, score_mode_a, STOP_WORDS

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DATASET_PATH = (
    r"C:\Users\hnrkh\.cache\huggingface\hub"
    r"\datasets--Percena--locomo-mc10"
    r"\snapshots\7d59a0463d83f97b042684310c0b3d17553004cd"
    r"\data\locomo_mc10.json"
)

N_QUESTIONS = 10  # per type


def load_first_conversation(path: str):
    """Load dataset and return questions for the first conversation."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # All records for the first conversation
    conv_id = records[0]["question_id"].rsplit("_", 1)[0]
    conv_qs = [r for r in records if r["question_id"].rsplit("_", 1)[0] == conv_id]
    return conv_id, conv_qs


def keyword_tokens(text: str) -> set[str]:
    """Extract meaningful keyword tokens from text, stripping stop words."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return words - STOP_WORDS


def check_retrieval_coverage(answer: str, recalled_chunks: list) -> dict:
    """Check whether the recalled text contains the correct answer's key words.

    Returns a dict with:
      - answer_keywords: set of non-stopword tokens from the answer
      - found_keywords: set of those tokens found anywhere in recalled text
      - coverage: fraction found
      - chunks_with_hits: how many of the top-20 chunks contain >= 1 keyword
      - best_chunk_coverage: best single-chunk coverage of answer keywords
      - best_chunk_idx: index of that chunk (0-based)
    """
    answer_kw = keyword_tokens(answer)
    if not answer_kw:
        return {
            "answer_keywords": answer_kw,
            "found_keywords": set(),
            "coverage": 1.0,  # trivially covered
            "chunks_with_hits": 0,
            "best_chunk_coverage": 1.0,
            "best_chunk_idx": -1,
        }

    all_recalled = ""
    chunks_with_hits = 0
    best_chunk_cov = 0.0
    best_chunk_idx = -1

    for i, r in enumerate(recalled_chunks):
        chunk_text = r.record.content.lower()
        all_recalled += " " + chunk_text
        chunk_kw = set(re.findall(r"[a-z0-9]+", chunk_text))
        hits = answer_kw & chunk_kw
        cov = len(hits) / len(answer_kw)
        if hits:
            chunks_with_hits += 1
        if cov > best_chunk_cov:
            best_chunk_cov = cov
            best_chunk_idx = i

    all_recalled_kw = set(re.findall(r"[a-z0-9]+", all_recalled))
    found = answer_kw & all_recalled_kw
    coverage = len(found) / len(answer_kw) if answer_kw else 1.0

    return {
        "answer_keywords": answer_kw,
        "found_keywords": found,
        "coverage": coverage,
        "chunks_with_hits": chunks_with_hits,
        "best_chunk_coverage": best_chunk_cov,
        "best_chunk_idx": best_chunk_idx,
    }


def diagnose_question(q: dict, results: list, verbose: bool = True) -> dict:
    """Run full diagnostic on one question. Returns summary dict."""
    question = q["question"]
    answer = q["answer"]
    correct_idx = q["correct_choice_index"]
    correct_choice = q["choices"][correct_idx]

    # Check retrieval coverage for the ground-truth answer text
    cov = check_retrieval_coverage(answer, results)

    # Also check coverage for the full correct choice text (sometimes more descriptive)
    cov_choice = check_retrieval_coverage(correct_choice, results)

    # Score breakdown
    top_scores = []
    for i, r in enumerate(results[:5]):
        top_scores.append({
            "rank": i + 1,
            "final_score": round(r.score, 5),
            "vec_score": round(r.vec_score, 4) if r.vec_score is not None else None,
            "fts_rank": round(r.fts_rank, 4) if r.fts_rank is not None else None,
        })

    # Run Mode-A scoring to see what it predicts
    recalled_text = ""
    for r in results:
        recalled_text += r.record.content[:1000] + "\n---\n"
    predicted = score_mode_a(question, q["choices"], recalled_text)
    is_correct = predicted == correct_idx

    # Determine failure mode
    if cov["coverage"] >= 0.8:
        if is_correct:
            failure_mode = "OK"
        else:
            failure_mode = "SCORING_FAIL"  # content is there, scoring missed
    elif cov["coverage"] >= 0.4:
        failure_mode = "PARTIAL_RETRIEVAL"
    else:
        failure_mode = "RETRIEVAL_FAIL"  # content not found at all

    diag = {
        "qid": q["question_id"],
        "type": q["question_type"],
        "question": question,
        "answer": answer,
        "correct_choice": correct_choice,
        "predicted_idx": predicted,
        "correct_idx": correct_idx,
        "is_correct": is_correct,
        "answer_coverage": round(cov["coverage"], 3),
        "choice_coverage": round(cov_choice["coverage"], 3),
        "answer_kw": cov["answer_keywords"],
        "found_kw": cov["found_keywords"],
        "missing_kw": cov["answer_keywords"] - cov["found_keywords"],
        "chunks_with_hits": cov["chunks_with_hits"],
        "best_chunk_cov": round(cov["best_chunk_coverage"], 3),
        "best_chunk_idx": cov["best_chunk_idx"],
        "top_scores": top_scores,
        "failure_mode": failure_mode,
        "total_results": len(results),
    }

    if verbose:
        print_diagnostic(diag)

    return diag


def print_diagnostic(d: dict):
    """Pretty-print one diagnostic result."""
    status = "CORRECT" if d["is_correct"] else "WRONG"
    print(f"\n{'='*70}")
    print(f"[{d['type']}] {d['qid']}  -- {status} -- {d['failure_mode']}")
    print(f"  Question:  {d['question']}")
    print(f"  Answer:    {d['answer']}")
    print(f"  Choice:    {d['correct_choice']}")
    print(f"  Predicted: idx={d['predicted_idx']}  Correct: idx={d['correct_idx']}")
    print(f"  Answer keyword coverage: {d['answer_coverage']*100:.0f}%  "
          f"(found {len(d['found_kw'])}/{len(d['answer_kw'])} keywords)")
    if d["missing_kw"]:
        print(f"  Missing keywords: {d['missing_kw']}")
    print(f"  Choice text coverage:   {d['choice_coverage']*100:.0f}%")
    print(f"  Chunks with any hit: {d['chunks_with_hits']}/{d['total_results']}")
    print(f"  Best single-chunk coverage: {d['best_chunk_cov']*100:.0f}% (rank #{d['best_chunk_idx']+1 if d['best_chunk_idx']>=0 else 'N/A'})")
    print(f"  Top-5 retrieval scores:")
    for s in d["top_scores"]:
        vec = f"vec={s['vec_score']}" if s["vec_score"] is not None else "vec=N/A"
        fts = f"fts={s['fts_rank']}" if s["fts_rank"] is not None else "fts=N/A"
        print(f"    #{s['rank']:2d}  final={s['final_score']:.5f}  {vec}  {fts}")


def main():
    print("NeuroPack LoCoMo Diagnostic")
    print("=" * 70)
    print(f"Dataset: {DATASET_PATH}")

    conv_id, conv_qs = load_first_conversation(DATASET_PATH)
    print(f"Conversation: {conv_id} ({len(conv_qs)} total questions)")

    # Split by type
    multi_hop = [q for q in conv_qs if q["question_type"] == "multi_hop"][:N_QUESTIONS]
    temporal = [q for q in conv_qs if q["question_type"] == "temporal_reasoning"][:N_QUESTIONS]
    print(f"Selected: {len(multi_hop)} multi_hop, {len(temporal)} temporal_reasoning")

    # Create store with same config as benchmark
    tmp_dir = tempfile.mkdtemp(prefix="np_diag_")
    db_path = os.path.join(tmp_dir, "diag.db")

    config = NeuropackConfig(
        db_path=db_path,
        embedder_type="sentence-transformer",
        embedding_dim=384,
        retrieval_weight_vec=0.6,
        retrieval_weight_fts=0.4,
        retrieval_weight_graph=0.0,
        recall_limit=20,
        auto_tag=False,
        dedup_threshold=0.99,
        reranker="off",
    )
    store = MemoryStore(config)
    store.initialize()

    # Ingest conversation
    sample = conv_qs[0]
    print(f"\nIngesting {len(sample['haystack_sessions'])} sessions "
          f"(8-turn sliding window, 50% overlap)...")
    t0 = time.perf_counter()
    ingest_conversation(
        store,
        sample["haystack_sessions"],
        sample.get("haystack_session_datetimes"),
    )
    t_ingest = time.perf_counter() - t0
    print(f"Ingestion done in {t_ingest:.2f}s")

    # ---------------------------------------------------------------
    # Run diagnostics
    # ---------------------------------------------------------------
    all_diags: list[dict] = []

    for label, questions in [("MULTI_HOP", multi_hop), ("TEMPORAL_REASONING", temporal)]:
        print(f"\n{'#'*70}")
        print(f"# {label} ({len(questions)} questions)")
        print(f"{'#'*70}")

        for q in questions:
            results = store.recall(q["question"], limit=20)
            diag = diagnose_question(q, results, verbose=True)
            all_diags.append(diag)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for qtype in ["multi_hop", "temporal_reasoning"]:
        diags = [d for d in all_diags if d["type"] == qtype]
        if not diags:
            continue

        n_correct = sum(1 for d in diags if d["is_correct"])
        n_total = len(diags)
        avg_cov = sum(d["answer_coverage"] for d in diags) / n_total
        avg_choice_cov = sum(d["choice_coverage"] for d in diags) / n_total
        avg_chunks = sum(d["chunks_with_hits"] for d in diags) / n_total

        # Count failure modes
        from collections import Counter
        modes = Counter(d["failure_mode"] for d in diags)

        print(f"\n{qtype}:")
        print(f"  Accuracy: {n_correct}/{n_total} ({n_correct/n_total*100:.0f}%)")
        print(f"  Avg answer keyword coverage: {avg_cov*100:.1f}%")
        print(f"  Avg choice text coverage:    {avg_choice_cov*100:.1f}%")
        print(f"  Avg chunks with hits:        {avg_chunks:.1f} / 20")
        print(f"  Failure modes:")
        for mode, count in modes.most_common():
            print(f"    {mode}: {count}")

    # Overall verdict
    retrieval_fails = sum(1 for d in all_diags if d["failure_mode"] == "RETRIEVAL_FAIL")
    partial_retrieval = sum(1 for d in all_diags if d["failure_mode"] == "PARTIAL_RETRIEVAL")
    scoring_fails = sum(1 for d in all_diags if d["failure_mode"] == "SCORING_FAIL")
    ok_count = sum(1 for d in all_diags if d["failure_mode"] == "OK")

    print(f"\nOVERALL DIAGNOSIS ({len(all_diags)} questions):")
    print(f"  OK (correct):          {ok_count}")
    print(f"  RETRIEVAL_FAIL:        {retrieval_fails}  (answer keywords < 40% covered)")
    print(f"  PARTIAL_RETRIEVAL:     {partial_retrieval}  (40-80% keyword coverage)")
    print(f"  SCORING_FAIL:          {scoring_fails}  (>= 80% keywords found, but wrong answer)")

    if retrieval_fails + partial_retrieval > scoring_fails:
        print("\n  --> PRIMARY BOTTLENECK: RETRIEVAL")
        print("      The search engine is not surfacing chunks that contain the answer.")
    elif scoring_fails > retrieval_fails + partial_retrieval:
        print("\n  --> PRIMARY BOTTLENECK: SCORING (Mode-A text matching)")
        print("      The content is retrieved but the TF-IDF matcher picks the wrong choice.")
    else:
        print("\n  --> MIXED: Both retrieval and scoring contribute roughly equally.")

    store.close()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
