#!/usr/bin/env python3
"""Build a BM25 index over PubMedQA documents.

This mirrors the SQuAD build script, but flattens each PubMedQA example into one
retrievable document by joining its evidence contexts into a single passage.

Usage:
  python FinalProject_working/BM25/01_build_pubmedqa_index.py
  python FinalProject_working/BM25/01_build_pubmedqa_index.py --pubmedqa data/PubMedQA/data/ori_pqal.json --index FinalProject_working/BM25/bm25_pubmedqa_index.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bm25_common import BM25Retriever, default_pubmedqa_index_path, default_pubmedqa_path, load_pubmedqa_documents


def main() -> int:
    p = argparse.ArgumentParser(description="Build BM25 index for PubMedQA")
    p.add_argument("--pubmedqa", type=Path, default=default_pubmedqa_path(), help="Path to PubMedQA JSON")
    p.add_argument("--index", type=Path, default=default_pubmedqa_index_path(), help="Where to write the BM25 index (.pkl)")
    args = p.parse_args()

    docs, _qas = load_pubmedqa_documents(args.pubmedqa)
    bm25 = BM25Retriever()
    bm25.add_documents(docs)
    bm25.save(args.index)

    print(f"Built BM25 index: {args.index}")
    print(f"Docs (PubMedQA items): {bm25.num_docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
