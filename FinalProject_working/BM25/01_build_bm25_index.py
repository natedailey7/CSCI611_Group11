#!/usr/bin/env python3
"""Task 1: Build a BM25 index over SQuAD v2 dev paragraphs.

Parameters are hard-coded in FinalProject_working/bm25_common.py.

Usage:
  python FinalProject_working/BM25/01_build_bm25_index.py
  python FinalProject_working/BM25/01_build_bm25_index.py --squad data/SQuAD/dev-v2.0.json --index FinalProject_working/BM25/bm25.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bm25_common import BM25Retriever, default_index_path, default_squad_path, load_squad_v2_paragraphs


def main() -> int:
    p = argparse.ArgumentParser(description="Build BM25 index for SQuAD v2 dev")
    p.add_argument("--squad", type=Path, default=default_squad_path(), help="Path to SQuAD v2 JSON")
    p.add_argument("--index", type=Path, default=default_index_path(), help="Where to write the BM25 index (.pkl)")
    args = p.parse_args()

    docs, _qas = load_squad_v2_paragraphs(args.squad)
    bm25 = BM25Retriever()
    bm25.add_documents(docs)
    bm25.save(args.index)

    print(f"Built BM25 index: {args.index}")
    print(f"Docs (paragraphs): {bm25.num_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
