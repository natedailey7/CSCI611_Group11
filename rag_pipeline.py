import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import nltk
from nltk.tokenize import sent_tokenize

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

OUTPUT_CSV = Path(r"outputs/rag_eval_results_pipeline_3.csv")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

K = 3              # number of retrieved chunks
N_HOLDOUT = 3      # number of items to hold out for out_domain evaluation
N_MISLEADING = 6   # total items for misleading context group (3 source + 3 eval)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────

print("Loading nvidia/TechQA-RAG-Eval...")
qa_dataset = load_dataset(
    "nvidia/TechQA-RAG-Eval",
    split="train"
)
print(f"Loaded {len(qa_dataset)} QA examples")

# ─────────────────────────────────────────────
# STEP 2: Normalize Dataset Format
# ─────────────────────────────────────────────

def normalize_item(item):
    return {
        "question":      item["question"],
        "answer":        str(item["answer"]),
        "passage_ids":   [ctx["filename"] for ctx in item["contexts"]],
        "passage_texts": {ctx["filename"]: ctx["text"] for ctx in item["contexts"]},
        "is_impossible": item["is_impossible"],
    }


all_data = [normalize_item(x) for x in qa_dataset]

# Select holdout indices from answerable items only so we always get N_HOLDOUT out_domain questions
answerable_indices = [i for i, item in enumerate(all_data) if not item["is_impossible"]]
HOLDOUT_INDICES = set(answerable_indices[:N_HOLDOUT])

# Select misleading group from the last N_MISLEADING non-holdout answerable items
non_holdout_answerable = [i for i in answerable_indices if i not in HOLDOUT_INDICES]
misleading_6 = non_holdout_answerable[-N_MISLEADING:]
A_trio_indices = misleading_6[:3]        # source passages: stay in corpus, provide wrong context
B_trio_indices = set(misleading_6[3:])   # eval questions: correct passages excluded from corpus

# Pair each B_trio question with its corresponding A_trio passage source (A[i] → B[i])
misleading_pairs = [
    (all_data[A_trio_indices[i]], all_data[misleading_6[3 + i]])
    for i in range(3)
]

holdout_data = [all_data[i] for i in sorted(HOLDOUT_INDICES)]
# corpus_data excludes holdout AND B_trio (B_trio correct passages must not be retrievable)
corpus_data  = [all_data[i] for i in range(len(all_data)) if i not in HOLDOUT_INDICES and i not in B_trio_indices]

# Build a global passage store: filename -> text (from all items)
all_passage_store = {}
for item in all_data:
    all_passage_store.update(item["passage_texts"])

# Passage IDs referenced by held-out questions (out_domain)
holdout_passage_ids = set()
for item in holdout_data:
    holdout_passage_ids.update(item["passage_ids"])

# Passage IDs referenced by B_trio (misleading_context eval questions)
b_trio_passage_ids = set()
for i in B_trio_indices:
    b_trio_passage_ids.update(all_data[i]["passage_ids"])

# Passage IDs present in the regular corpus (includes A_trio, which stays in corpus)
corpus_passage_ids = set()
for item in corpus_data:
    corpus_passage_ids.update(item["passage_ids"])

# Exclude passages exclusive to holdout or B_trio; A_trio passages stay because A_trio is in corpus_data
excluded_ids = (holdout_passage_ids | b_trio_passage_ids) - corpus_passage_ids

print(f"Excluded passage IDs (holdout + B_trio exclusive): {len(excluded_ids)}")

# ─────────────────────────────────────────────
# STEP 3: Context Chunking
# ─────────────────────────────────────────────

def chunk_text(text, max_words=250, overlap=50):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_words)]
            current_len = len(overlap_words)

        current_chunk.append(sentence)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Build corpus chunks from all passages EXCEPT those exclusive to holdout items
corpus_chunks = []
for pid, text in all_passage_store.items():
    if pid not in excluded_ids:
        corpus_chunks.extend(chunk_text(text))

print(f"Corpus chunks: {len(corpus_chunks)}")

# ─────────────────────────────────────────────
# STEP 4: Load Language Model
# ─────────────────────────────────────────────

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model (this will take time)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("Model loaded successfully")

# ─────────────────────────────────────────────
# STEP 5: Build BM25 Retriever
# ─────────────────────────────────────────────

corpus_tokenized = [doc.split() for doc in corpus_chunks]
bm25 = BM25Okapi(corpus_tokenized)
print("BM25 ready")

# ─────────────────────────────────────────────
# STEP 6: Build Dense Retriever (FAISS + MiniLM)
# ─────────────────────────────────────────────

print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding corpus chunks...")
corpus_embeddings = embed_model.encode(corpus_chunks, convert_to_numpy=True)

dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(corpus_embeddings)
print("FAISS index ready")

# ─────────────────────────────────────────────
# RETRIEVAL HELPERS
# ─────────────────────────────────────────────

def retrieve_bm25_with_scores(query, k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    seen = set()
    for idx in sorted_indices:
        text = corpus_chunks[idx]
        if text not in seen:
            results.append((text, float(scores[idx])))
            seen.add(text)
        if len(results) == k:
            break
    return results


def retrieve_dense_with_scores(query, k=5):
    q_emb = embed_model.encode([query])
    D, I = index.search(q_emb, k * 3)

    results = []
    seen = set()
    for dist, idx in zip(D[0], I[0]):
        text = corpus_chunks[idx]
        if text not in seen:
            results.append((text, float(dist)))
            seen.add(text)
        if len(results) == k:
            break
    return results


def split_contexts_and_scores(context_score_pairs):
    contexts = [x[0] for x in context_score_pairs]
    scores   = [x[1] for x in context_score_pairs]
    return contexts, scores

# ─────────────────────────────────────────────
# GENERATION HELPERS
# ─────────────────────────────────────────────

def build_prompt(contexts, question):
    context_text = "\n\n".join(contexts)
    return (
        "Use the context provided to answer the question.\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_no_corpus_prompt(question):
    return f"Question: {question}\n\nAnswer:"


def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=125,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    print(f"  Generation took {(time.time() - start):.1f}s")
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result.split("Answer:")[-1].strip()
    return answer

# ─────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def exact_match(pred, truth):
    return int(normalize(pred) == normalize(truth))


def f1_score(pred, truth):
    pred_tokens  = normalize(pred).split()
    truth_tokens = normalize(truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_in_corpus(answer, chunks):
    if not chunks:
        return False
    return any(answer.lower() in c.lower() for c in chunks)


def make_context_columns(contexts, scores=None, k=3):
    context_dict = {}
    if scores is None:
        scores = []
    for i in range(k):
        context_dict[f"context_{i+1}"]       = contexts[i] if i < len(contexts) else ""
        context_dict[f"context_{i+1}_score"] = scores[i]   if i < len(scores)   else ""
    return context_dict


def run_eval_item(item, retriever_name, contexts=None, scores=None, k=3):
    if contexts is None:
        contexts = []
    if scores is None:
        scores = []

    if retriever_name == "No Corpus":
        prompt = build_no_corpus_prompt(item["question"])
    else:
        prompt = build_prompt(contexts, item["question"])

    pred = generate_answer(prompt)

    row = {
        "group":                    item["group"],
        "retriever":                retriever_name,
        "question":                 item["question"],
        "ground_truth":             item["ground_truth"],
        "prediction":               pred,
        "f1":                       f1_score(pred, item["ground_truth"]),
        "em":                       exact_match(pred, item["ground_truth"]),
        "answer_in_retrieved":      answer_in_corpus(item["ground_truth"], contexts),
        "answer_in_any_corpus_chunk": answer_in_corpus(item["ground_truth"], corpus_chunks),
        "combined_retrieved_context": "\n\n---\n\n".join(contexts),
    }
    row.update(make_context_columns(contexts, scores=scores, k=k))
    return row

# ─────────────────────────────────────────────
# STEP 7: Build Eval Questions
# ─────────────────────────────────────────────

# holdout_data is already all answerable (selected above)
# corpus_data may still contain impossible items — filter for in_domain eval
answerable_corpus = [item for item in corpus_data if not item["is_impossible"]]

# In-domain: first 3 answerable items from corpus_data (their passages ARE in the corpus)
in_domain_questions = [
    {
        "group":        "in_domain",
        "question":     answerable_corpus[i]["question"],
        "ground_truth": answerable_corpus[i]["answer"],
    }
    for i in range(3)
]

# Out-of-domain: the N_HOLDOUT held-out items (their passages are NOT in the corpus)
out_domain_questions = [
    {
        "group":        "out_domain",
        "question":     item["question"],
        "ground_truth": item["answer"],
    }
    for item in holdout_data
]

# Misleading context: B_trio questions with A_trio's passages forced as context.
# The corpus contains A_trio's passages (wrong answers) but not B_trio's (correct answers).
misleading_questions = [
    {
        "group":           "misleading_context",
        "question":        b_item["question"],
        "ground_truth":    b_item["answer"],
        "forced_contexts": list(a_item["passage_texts"].values()),
    }
    for a_item, b_item in misleading_pairs
]

eval_questions_standard   = in_domain_questions + out_domain_questions
eval_questions_misleading = misleading_questions

# ─────────────────────────────────────────────
# STEP 8: Run Evaluation
# ─────────────────────────────────────────────

rows = []

# Standard evaluation (in_domain + out_domain)
for item in eval_questions_standard:
    q = item["question"]
    print(f"\nRunning: {item['group']} | {q}")

    bm25_pairs  = retrieve_bm25_with_scores(q, k=K)
    dense_pairs = retrieve_dense_with_scores(q, k=K)

    bm25_contexts,  bm25_scores  = split_contexts_and_scores(bm25_pairs)
    dense_contexts, dense_scores = split_contexts_and_scores(dense_pairs)

    rows.append(run_eval_item(item, "BM25",      bm25_contexts,  scores=bm25_scores,  k=K))
    rows.append(run_eval_item(item, "Dense",     dense_contexts, scores=dense_scores, k=K))
    rows.append(run_eval_item(item, "No Corpus", contexts=[],    scores=[],           k=K))

# Misleading context evaluation (appended at end of CSV)
# Each question runs BM25/Dense/No Corpus (for comparison) plus a forced "Misleading" retriever
# that injects A_trio's passage as context — the corpus has A's answer, not B's
for item in eval_questions_misleading:
    q = item["question"]
    print(f"\nRunning: {item['group']} | {q}")

    bm25_pairs  = retrieve_bm25_with_scores(q, k=K)
    dense_pairs = retrieve_dense_with_scores(q, k=K)

    bm25_contexts,  bm25_scores  = split_contexts_and_scores(bm25_pairs)
    dense_contexts, dense_scores = split_contexts_and_scores(dense_pairs)

    forced_contexts = item["forced_contexts"][:K]

    rows.append(run_eval_item(item, "BM25",       bm25_contexts,  scores=bm25_scores,  k=K))
    rows.append(run_eval_item(item, "Dense",       dense_contexts, scores=dense_scores, k=K))
    rows.append(run_eval_item(item, "No Corpus",   contexts=[],    scores=[],           k=K))
    rows.append(run_eval_item(item, "Misleading",  forced_contexts, scores=[],          k=K))

# ─────────────────────────────────────────────
# STEP 9: Save Results
# ─────────────────────────────────────────────

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved results to: {OUTPUT_CSV}")
print(results_df[[
    "group", "retriever", "question", "ground_truth", "f1", "em", "answer_in_retrieved"
]].to_string())
