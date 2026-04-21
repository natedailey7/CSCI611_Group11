#!/usr/bin/env python3
"""Task 3: Run BM25 retrieval for every SQuAD dev question + (optionally) report recall.

Hard-coded parameters (K1/B/TOPK) live in FinalProject_working/bm25_common.py.

Usage:
  # Build the index once (recommended)
    python FinalProject_working/BM25/01_build_bm25_index.py

  # Produce per-question JSONL with retrieved paragraphs
  python FinalProject_working/03_run_dev_retrieval.py --output data/SQuAD/bm25_topk.jsonl

  # Also print Recall@TOPK and MRR@TOPK
  python FinalProject_working/03_run_dev_retrieval.py --output data/SQuAD/bm25_topk.jsonl --report-recall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from bm25_common import TOPK, BM25Retriever, compute_recall_mrr_at_k, default_index_path, default_squad_path, load_squad_v2_paragraphs


def main() -> int:
    p = argparse.ArgumentParser(description="Run BM25 retrieval over all SQuAD dev questions")
    p.add_argument("--squad", type=Path, default=default_squad_path(), help="Path to SQuAD v2 JSON")
    p.add_argument("--index", type=Path, default=default_index_path(), help="BM25 index path (.pkl)")
    p.add_argument("--output", type=Path, required=True, help="Write JSONL results to this path")
    p.add_argument("--report-recall", action="store_true", help="Print Recall@TOPK and MRR@TOPK")
    args = p.parse_args()

    if not args.index.exists():
        raise SystemExit(
            f"Index not found: {args.index}. Build it first with: python FinalProject_working/BM25/01_build_bm25_index.py"
        )

    docs, qas = load_squad_v2_paragraphs(args.squad)
    bm25 = BM25Retriever.load(args.index)

    # Safety: ensure index and docs align by doc_id range.
    # (If you rebuilt the index from the same SQuAD file, this should match.)
    if bm25.num_docs != len(docs):
        print(
            f"WARNING: index has {bm25.num_docs} docs but SQuAD parse has {len(docs)} docs. "
            "If results look odd, rebuild the index from the same SQuAD file.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_f:
        for qa in qas:
            qid = qa.get("id")
            question = qa.get("question")
            gold_doc_id = qa.get("context_doc_id")
            if not isinstance(question, str) or not isinstance(gold_doc_id, int):
                continue

            retrieved = bm25.retrieve(question, topk=TOPK)

            record: Dict[str, Any] = {
                "id": qid,
                "question": question,
                "gold_doc_id": gold_doc_id,
                "topk": [
                    {
                        "doc_id": doc.doc_id,
                        "score": score,
                        "title": doc.title,
                        "para_id": doc.para_id,
                        "context": doc.context,
                    }
                    for doc, score in retrieved
                ],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.output}")

    if args.report_recall:
        recall, mrr, hit_q, total_q = compute_recall_mrr_at_k(bm25, qas, topk=TOPK)
        print(f"BM25 paragraph Recall@{TOPK}: {recall:.4f} ({hit_q}/{total_q})")
        print(f"BM25 paragraph MRR@{TOPK}: {mrr:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
