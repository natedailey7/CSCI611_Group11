"""Shared BM25 + SQuAD(v2) utilities.

Hard-coded defaults (edit here if you want to tune BM25):
- K1, B: BM25 parameters
- TOPK: retrieval depth used by the task scripts

No third-party dependencies.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

# ---- Hard-coded parameters (requested) ----
K1: float = 1.2
B: float = 0.75
TOPK: int = 5

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

_THIS_DIR = Path(__file__).resolve().parent
# Expected layout: <repo>/FinalProject_working/BM25/bm25_common.py
_PROJECT_ROOT = _THIS_DIR.parents[1]


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class SquadDoc:
    doc_id: int
    title: str
    para_id: int
    context: str


class BM25Retriever:
    """BM25 Okapi retriever with an inverted index."""

    def __init__(self, *, k1: float = K1, b: float = B) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)

        self._docs: List[SquadDoc] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._N: int = 0

        # term -> list of (doc_id, tf)
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        # term -> idf
        self._idf: Dict[str, float] = {}

    @property
    def num_docs(self) -> int:
        return self._N

    @property
    def docs(self) -> Sequence[SquadDoc]:
        return self._docs

    def add_documents(self, docs: Sequence[SquadDoc]) -> None:
        self._docs = list(docs)
        self._N = len(self._docs)
        if self._N == 0:
            raise ValueError("No documents provided")

        postings: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()
        doc_len: List[int] = [0] * self._N

        for doc in self._docs:
            terms = tokenize(doc.context)
            doc_len[doc.doc_id] = len(terms)
            tf = Counter(terms)
            for term, freq in tf.items():
                postings[term].append((doc.doc_id, freq))
            for term in tf.keys():
                doc_freq[term] += 1

        self._doc_len = doc_len
        self._avgdl = sum(doc_len) / self._N
        self._postings = dict(postings)

        idf: Dict[str, float] = {}
        for term, df in doc_freq.items():
            # Stabilized Robertson/Sparck Jones IDF
            idf[term] = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))
        self._idf = idf

    def score_query(self, query: str) -> Dict[int, float]:
        q_terms = tokenize(query)
        if not q_terms:
            return {}

        q_unique = list(dict.fromkeys(q_terms))
        scores: DefaultDict[int, float] = defaultdict(float)

        for term in q_unique:
            idf = self._idf.get(term)
            if idf is None:
                continue
            postings = self._postings.get(term)
            if not postings:
                continue
            for doc_id, tf in postings:
                dl = self._doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0) / denom)

        return dict(scores)

    def retrieve(self, query: str, *, topk: int = TOPK) -> List[Tuple[SquadDoc, float]]:
        if topk <= 0:
            raise ValueError("topk must be > 0")

        scores = self.score_query(query)
        if not scores:
            return []

        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        return [(self._docs[doc_id], score) for doc_id, score in best]

    def save(self, path: Path) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "docs": self._docs,
            "doc_len": self._doc_len,
            "avgdl": self._avgdl,
            "N": self._N,
            "postings": self._postings,
            "idf": self._idf,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        with path.open("rb") as f:
            payload = pickle.load(f)

        bm25 = cls(k1=payload["k1"], b=payload["b"])
        bm25._docs = payload["docs"]
        bm25._doc_len = payload["doc_len"]
        bm25._avgdl = payload["avgdl"]
        bm25._N = payload["N"]
        bm25._postings = payload["postings"]
        bm25._idf = payload["idf"]
        return bm25


def load_squad_v2_paragraphs(squad_path: Path) -> Tuple[List[SquadDoc], List[Dict[str, Any]]]:
    """Return (docs, qas).

    - docs: paragraph contexts turned into SquadDoc with stable doc_id
    - qas: list of {id, question, context_doc_id, ...}

    Gold for retrieval is the *paragraph context* that the question came from.
    """

    with squad_path.open("r", encoding="utf-8") as f:
        squad = json.load(f)

    data = squad.get("data")
    if not isinstance(data, list):
        raise ValueError("Invalid SQuAD file: missing 'data' list")

    docs: List[SquadDoc] = []
    qas: List[Dict[str, Any]] = []

    doc_id = 0
    for article in data:
        title = str(article.get("title", ""))
        paragraphs = article.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue

        for para_id, para in enumerate(paragraphs):
            context = para.get("context")
            if not isinstance(context, str) or not context.strip():
                continue

            docs.append(SquadDoc(doc_id=doc_id, title=title, para_id=para_id, context=context))

            qa_items = para.get("qas")
            if isinstance(qa_items, list):
                for qa in qa_items:
                    if not isinstance(qa, dict):
                        continue
                    qas.append(
                        {
                            "id": qa.get("id"),
                            "question": qa.get("question"),
                            "title": title,
                            "para_id": para_id,
                            "context_doc_id": doc_id,
                        }
                    )

            doc_id += 1

    if not docs:
        raise ValueError("No paragraphs found in SQuAD file")

    return docs, qas


def compute_recall_mrr_at_k(
    bm25: BM25Retriever,
    qas: Sequence[Dict[str, Any]],
    *,
    topk: int = TOPK,
) -> Tuple[float, float, int, int]:
    total_q = 0
    hit_q = 0
    rr_sum = 0.0

    for qa in qas:
        question = qa.get("question")
        gold_doc_id = qa.get("context_doc_id")
        if not isinstance(question, str) or not isinstance(gold_doc_id, int):
            continue

        retrieved = bm25.retrieve(question, topk=topk)
        total_q += 1

        rank = None
        for i, (doc, _score) in enumerate(retrieved, start=1):
            if doc.doc_id == gold_doc_id:
                rank = i
                break

        if rank is not None:
            hit_q += 1
            rr_sum += 1.0 / rank

    recall = (hit_q / total_q) if total_q else 0.0
    mrr = (rr_sum / total_q) if total_q else 0.0
    return recall, mrr, hit_q, total_q


def default_squad_path() -> Path:
    return _PROJECT_ROOT / "data" / "SQuAD" / "dev-v2.0.json"


def default_index_path() -> Path:
    return _THIS_DIR / "bm25_squad_dev_index.pkl"
