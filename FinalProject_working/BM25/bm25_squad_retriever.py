#!/usr/bin/env python3
"""BM25 sparse retriever for SQuAD v2 dev (wrapper).

This file used to be an all-in-one script. It’s now a thin wrapper around the
split task scripts in this folder:

- FinalProject_working/01_build_bm25_index.py
- FinalProject_working/02_query_bm25.py
- FinalProject_working/03_run_dev_retrieval.py

(In this workspace those scripts live under FinalProject_working/BM25/.)

BM25 parameters (k1/b) and default TOPK are hard-coded in
FinalProject_working/bm25_common.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bm25_common import (
    TOPK,
    BM25Retriever,
    compute_recall_mrr_at_k,
    default_index_path,
    default_squad_path,
    load_squad_v2_paragraphs,
)


def build_index(squad_path: Path, index_path: Path) -> None:
    docs, _qas = load_squad_v2_paragraphs(squad_path)
    bm25 = BM25Retriever()
    bm25.add_documents(docs)
    bm25.save(index_path)


def run_query(index_path: Path, query: str) -> int:
    if not index_path.exists():
        raise SystemExit(
            f"Index not found: {index_path}. Build it first with: python FinalProject_working/BM25/01_build_bm25_index.py"
        )
    bm25 = BM25Retriever.load(index_path)
    hits = bm25.retrieve(query, topk=TOPK)
    if not hits:
        print("NO_HITS")
        return 0
    for rank, (doc, score) in enumerate(hits, start=1):
        snippet = doc.context.replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260] + "..."
        print(f"#{rank}\tscore={score:.4f}\tdoc_id={doc.doc_id}\ttitle={doc.title}\tpara={doc.para_id}")
        print(snippet)
        print()
    return 0


def run_dev(index_path: Path, squad_path: Path, output_path: Path, report_recall: bool) -> int:
    if not index_path.exists():
        raise SystemExit(
            f"Index not found: {index_path}. Build it first with: python FinalProject_working/BM25/01_build_bm25_index.py"
        )

    _docs, qas = load_squad_v2_paragraphs(squad_path)
    bm25 = BM25Retriever.load(index_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for qa in qas:
            qid = qa.get("id")
            question = qa.get("question")
            gold_doc_id = qa.get("context_doc_id")
            if not isinstance(question, str) or not isinstance(gold_doc_id, int):
                continue

            retrieved = bm25.retrieve(question, topk=TOPK)
            record = {
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

    print(f"Wrote: {output_path}")

    if report_recall:
        recall, mrr, hit_q, total_q = compute_recall_mrr_at_k(bm25, qas, topk=TOPK)
        print(f"BM25 paragraph Recall@{TOPK}: {recall:.4f} ({hit_q}/{total_q})")
        print(f"BM25 paragraph MRR@{TOPK}: {mrr:.4f}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="BM25 sparse retriever over SQuAD v2 dev (wrapper)")
    p.add_argument("--squad", type=Path, default=default_squad_path(), help="Path to SQuAD v2 JSON")
    p.add_argument("--index", type=Path, default=default_index_path(), help="BM25 index path (.pkl)")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-index", action="store_true", help="Build BM25 index")
    mode.add_argument("--query", type=str, help="Run a single query")
    mode.add_argument("--run-dev", action="store_true", help="Run retrieval for all dev questions")

    p.add_argument("--output", type=Path, default=Path("data/SQuAD/bm25_topk.jsonl"), help="Output JSONL")
    p.add_argument("--report-recall", action="store_true", help="Print Recall@TOPK and MRR@TOPK")
    args = p.parse_args()

    if args.build_index:
        build_index(args.squad, args.index)
        print(f"Built BM25 index: {args.index}")
        return 0

    if args.query is not None:
        return run_query(args.index, args.query)

    return run_dev(args.index, args.squad, args.output, args.report_recall)


if __name__ == "__main__":
    raise SystemExit(main())
