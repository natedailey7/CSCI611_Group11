#!/usr/bin/env python3
"""Task 4: Simple RAG using BM25 (sparse retriever) + GPT-2 (generator).

This script:
- Loads your existing BM25 index (built from SQuAD v2 dev paragraphs)
- Retrieves top-k paragraph contexts for a question
- Prompts a Hugging Face causal LM (default: GPT-2) with the retrieved context

Notes / caveats:
- GPT-2 is not instruction-tuned, so answer quality will be limited.
- SQuAD v2 includes unanswerable questions; this script does not implement abstention.

Examples:
  # Build index first (once)
  python FinalProject_working/BM25/01_build_bm25_index.py
    python FinalProject_working/BM25/01_build_pubmedqa_index.py

    # Run using the hard-coded settings below
    python FinalProject_working/BM25/04_rag_gpt2.py

    # To switch dataset, query list, or baseline vs corpus mode, edit the constants near the top of this file.

    # Run over the selected dataset's questions (write JSONL)
    # Set RUN_MODE = "run-dev", then run:
    python FinalProject_working/BM25/04_rag_gpt2.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bm25_common import (
    BM25Retriever,
    QADoc,
    default_index_path,
    default_pubmedqa_index_path,
    default_pubmedqa_path,
    default_squad_path,
    load_pubmedqa_documents,
    load_squad_v2_paragraphs,
)


# ---- Hard-coded parameters (edit here if you want to tune) ----
# Dataset / execution mode
DATASET: str = "squad"  # "squad" or "pubmedqa"
RUN_MODE: str = "query"  # "query" or "run-dev"
USE_CORPUS: bool = True

# Query mode
QUERY_TEXTS: List[str] = [
    "When was the Eiffel Tower built?",
    "What is the capital of France?",
    "Who wrote 'Pride and Prejudice'?",
    "When did the Apollo 11 mission land on the moon?",
]

# Paths
SQUAD_INDEX_PATH: Path = default_index_path()
PUBMEDQA_INDEX_PATH: Path = default_pubmedqa_index_path()
SQUAD_PATH: Path = default_squad_path()
PUBMEDQA_PATH: Path = default_pubmedqa_path()
SQUAD_OUTPUT_PATH: Path = Path("data/SQuAD/gpt2_rag_bm25.jsonl")
PUBMEDQA_OUTPUT_PATH: Path = Path("data/PubMedQA/gpt2_rag_bm25.jsonl")

# Retrieval
RAG_TOPK: int = 5

# Generation
MODEL_NAME: str = "gpt2"  # or "distilgpt2" for faster downloads
MAX_NEW_TOKENS: int = 64
DO_SAMPLE: bool = False
TEMPERATURE: float = 0.7
TOP_P: float = 0.9
SEED: Optional[int] = None

# Dev run
DEV_LIMIT: Optional[int] = None


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency for RAG script. Install with:\n"
            "  pip install transformers torch\n\n"
            f"Original import error: {e}"
        )


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_prompt(question: str, contexts: Sequence[str]) -> str:
    # Keep prompt format very explicit so GPT-2 has a consistent pattern.
    joined = "\n\n".join(f"[Context {i+1}] {c.strip()}" for i, c in enumerate(contexts) if c.strip())
    return (
        "You are a question answering system. Answer the question using ONLY the provided context.\n"
        "If the context does not contain the answer, say: unanswerable.\n\n"
        f"Question: {question.strip()}\n\n"
        f"{joined}\n\n"
        "Answer:"
    )


def _count_tokens(tokenizer: Any, text: str) -> int:
    tokenized = tokenizer(text, add_special_tokens=False, return_attention_mask=False, verbose=False)
    return len(tokenized["input_ids"])


def _truncate_contexts_to_fit(
    *,
    question: str,
    retrieved: Sequence[Tuple[QADoc, float]],
    tokenizer: Any,
    max_input_tokens: int,
) -> List[str]:
    """Greedily pack contexts until the prompt would exceed max_input_tokens."""

    contexts: List[str] = []
    for doc, _score in retrieved:
        candidate_contexts = contexts + [doc.context]
        prompt = _build_prompt(question, candidate_contexts)
        prompt_len = _count_tokens(tokenizer, prompt)
        if prompt_len <= max_input_tokens:
            contexts = candidate_contexts
        else:
            break

    if contexts:
        return contexts

    # Fallback: include a truncated first context if nothing fits.
    if not retrieved:
        return []

    first_context = retrieved[0][0].context
    # Binary-ish shrink: progressively cut characters until it fits.
    lo, hi = 0, len(first_context)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = first_context[:mid]
        prompt = _build_prompt(question, [candidate])
        if _count_tokens(tokenizer, prompt) <= max_input_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    return [best] if best else []


def _generate_answer(
    *,
    question: str,
    retrieved: Sequence[Tuple[QADoc, float]],
    use_corpus: bool,
) -> Dict[str, Any]:
    _require_transformers()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if SEED is not None:
        set_seed(SEED)

    device = _select_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # GPT-2 has no pad token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    # Leave room for generated tokens.
    n_positions = getattr(model.config, "n_positions", 1024)
    max_input_tokens = max(1, int(n_positions) - int(MAX_NEW_TOKENS) - 1)

    retrieved = list(retrieved)[:RAG_TOPK]
    packed_contexts: List[str] = []
    if use_corpus:
        packed_contexts = _truncate_contexts_to_fit(
            question=question, retrieved=retrieved, tokenizer=tokenizer, max_input_tokens=max_input_tokens
        )

    prompt = _build_prompt(question, packed_contexts)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "pad_token_id": tokenizer.eos_token_id,
    }

    if DO_SAMPLE:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": float(TEMPERATURE),
                "top_p": float(TOP_P),
            }
        )
    else:
        gen_kwargs.update({"do_sample": False})

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract everything after the final "Answer:" marker.
    answer = decoded.split("Answer:")[-1].strip()
    # Light cleanup: keep the first line and stop if the model starts emitting prompt-like markers.
    for marker in ("Question:", "[Context", "==="):
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()
    answer = answer.splitlines()[0].strip() if answer else answer

    return {
        "model": MODEL_NAME,
        "device": device,
        "question": question,
        "prompt": prompt,
        "answer": answer,
        "use_corpus": use_corpus,
        "retrieved": [
            {
                "doc": asdict(doc),
                "score": float(score),
            }
            for doc, score in retrieved
        ],
    }


def _load_bm25(index_path: Path) -> BM25Retriever:
    if not index_path.exists():
        raise SystemExit(
            f"Index not found: {index_path}. Build it first with: python FinalProject_working/BM25/01_build_bm25_index.py"
        )
    return BM25Retriever.load(index_path)


def _run_single_query(
    *,
    bm25: BM25Retriever,
    query: str,
    use_corpus: bool,
) -> int:
    retrieved: Sequence[Tuple[QADoc, float]] = []
    if use_corpus:
        retrieved = bm25.retrieve(query, topk=RAG_TOPK)

    if use_corpus and not retrieved:
        print("NO_HITS")
        return 0

    result = _generate_answer(
        question=query,
        retrieved=retrieved,
        use_corpus=use_corpus,
    )

    # Pretty console output.
    if use_corpus:
        print("=== Retrieved contexts (top-k) ===")
        for i, item in enumerate(result["retrieved"], start=1):
            doc = item["doc"]
            score = item["score"]
            snippet = doc["context"].replace("\n", " ")
            if len(snippet) > 260:
                snippet = snippet[:260] + "..."
            print(f"#{i}\tscore={score:.4f}\tdoc_id={doc['doc_id']}\ttitle={doc['title']}\tpara={doc['para_id']}")
            print(snippet)
            print()
    else:
        print("=== Mode ===")
        print("baseline GPT-2 (no retrieved corpus)")
        print()

    print("=== Generated answer ===")
    print(result["answer"])  # noqa: T201
    return 0


def _iter_dev_questions(dataset: str, *, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    if dataset == "squad":
        _docs, qas = load_squad_v2_paragraphs(SQUAD_PATH)
    elif dataset == "pubmedqa":
        _docs, qas = load_pubmedqa_documents(PUBMEDQA_PATH)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    n = 0
    for qa in qas:
        question = qa.get("question")
        qid = qa.get("id")
        if not isinstance(question, str) or not question.strip():
            continue
        yield {"id": qid, "question": question, "gold_doc_id": qa.get("context_doc_id")}
        n += 1
        if limit is not None and n >= limit:
            break


def _run_dev(
    *,
    bm25: BM25Retriever,
    output_path: Path,
    limit: Optional[int],
    use_corpus: bool,
    dataset: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for qa in _iter_dev_questions(dataset, limit=limit):
            qid = qa["id"]
            question = qa["question"]

            retrieved: Sequence[Tuple[QADoc, float]] = []
            if use_corpus:
                retrieved = bm25.retrieve(question, topk=RAG_TOPK)

            result = _generate_answer(
                question=question,
                retrieved=retrieved,
                use_corpus=use_corpus,
            )

            record: Dict[str, Any] = {
                "id": qid,
                "question": question,
                "gold_doc_id": qa.get("gold_doc_id"),
                "dataset": dataset,
                "model": result["model"],
                "device": result["device"],
                "use_corpus": result["use_corpus"],
                "topk": [
                    {
                        "doc_id": r["doc"]["doc_id"],
                        "score": r["score"],
                        "title": r["doc"]["title"],
                        "para_id": r["doc"]["para_id"],
                        "context": r["doc"]["context"],
                    }
                    for r in result["retrieved"]
                ],
                "answer": result["answer"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote: {output_path}")
    return 0


def main() -> int:
    if RAG_TOPK <= 0:
        raise SystemExit("RAG_TOPK must be > 0")
    if MAX_NEW_TOKENS <= 0:
        raise SystemExit("MAX_NEW_TOKENS must be > 0")
    if DATASET not in {"squad", "pubmedqa"}:
        raise SystemExit('DATASET must be either "squad" or "pubmedqa"')
    if RUN_MODE not in {"query", "run-dev"}:
        raise SystemExit('RUN_MODE must be either "query" or "run-dev"')

    index_path = SQUAD_INDEX_PATH if DATASET == "squad" else PUBMEDQA_INDEX_PATH
    output_path = SQUAD_OUTPUT_PATH if DATASET == "squad" else PUBMEDQA_OUTPUT_PATH
    bm25 = _load_bm25(index_path)

    if RUN_MODE == "query":
        queries = [query.strip() for query in QUERY_TEXTS if query.strip()]
        if not queries:
            raise SystemExit("QUERY_TEXTS must contain at least one non-empty query when RUN_MODE='query'")

        for index, query in enumerate(queries, start=1):
            if len(queries) > 1:
                print(f"=== Query {index}/{len(queries)} ===")
                print(query)
            _run_single_query(
                bm25=bm25,
                query=query,
                use_corpus=USE_CORPUS,
            )
            if index < len(queries):
                print()

        return 0

    return _run_dev(
        bm25=bm25,
        output_path=output_path,
        limit=DEV_LIMIT,
        use_corpus=USE_CORPUS,
        dataset=DATASET,
    )


if __name__ == "__main__":
    raise SystemExit(main())
