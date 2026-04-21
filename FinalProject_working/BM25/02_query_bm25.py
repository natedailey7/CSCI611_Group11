#!/usr/bin/env python3
"""Task 2: Load a BM25 index and run a single query.

BM25 parameters + default TOPK are hard-coded in FinalProject_working/bm25_common.py.

Usage:
    python FinalProject_working/BM25/02_query_bm25.py --query "When was the Eiffel Tower built?"
    python FinalProject_working/BM25/02_query_bm25.py --index FinalProject_working/BM25/bm25.pkl --query "..."

Notes:
- This returns top paragraphs (retrieval), not the extracted answer span.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bm25_common import TOPK, BM25Retriever, default_index_path


def main() -> int:
    p = argparse.ArgumentParser(description="Query BM25 index (SQuAD paragraphs)")
    p.add_argument("--index", type=Path, default=default_index_path(), help="BM25 index path (.pkl)")
    p.add_argument(
        "--query",
        nargs="+",
        required=True,
        help="Query text. You can either quote it (recommended) or pass multiple words without quotes.",
    )
    args = p.parse_args()

    query = " ".join(args.query).strip()

    if not args.index.exists():
        raise SystemExit(
            f"Index not found: {args.index}. Build it first with: python FinalProject_working/BM25/01_build_bm25_index.py"
        )

    bm25 = BM25Retriever.load(args.index)
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


if __name__ == "__main__":
    raise SystemExit(main())
