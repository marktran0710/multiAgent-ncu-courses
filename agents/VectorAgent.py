# ─────────────────────────────────────────────────────────────────────────────
#  Agent 3 — VectorAgent
# ─────────────────────────────────────────────────────────────────────────────


import math

import chromadb
from chromadb.config import Settings

from function.main import tokenize
from models.Course import Course
from models.RetrievalResult import RetrievalResult
from models.UserProfile import UserProfile



try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    print("Optional: pip install sentence-transformers for better local embeddings")
    _ST_AVAILABLE = False


class VectorAgent:
    """
    Semantic retrieval using sentence-transformers embeddings stored in
    ChromaDB.  Falls back to in-memory TF-IDF cosine similarity when
    sentence-transformers is not installed.
    """

    name = "VectorAgent"
    COLLECTION = "ncu_courses"  # ChromaDB collection name

    def __init__(self, courses: list[Course]):
        self.courses = courses
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

        if _ST_AVAILABLE:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_transformer = True
        else:
            self._use_transformer = False

        self._build_index(courses)

    def _embed(self, text: str) -> list[float]:
        if self._use_transformer:
            return self.embed_model.encode(text).tolist()
        raise RuntimeError("SentenceTransformer not available")

    def _build_index(self, courses: list[Course]):
        if self._use_transformer:
            print(f"[{self.name}] Building vector index (Sentence-Transformers) …")
            try:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.COLLECTION, metadata={"hnsw:space": "cosine"}
                )
                existing = set(self.collection.get()["ids"])
                to_add = [c for c in courses if c.id not in existing]
                if to_add:
                    self.collection.upsert(
                        ids=[c.id for c in to_add],
                        embeddings=[self._embed(c.full_text()) for c in to_add],
                        documents=[c.full_text() for c in to_add],
                        metadatas=[{"name": c.name} for c in to_add],
                    )
                    print(f"[{self.name}] Upserted {len(to_add)} embeddings.")
                else:
                    print(f"[{self.name}] Embeddings already cached.")
                self._index_ok = True
                return
            except Exception as exc:
                print(f"[{self.name}] WARNING: ChromaDB/embedding error ({exc}). Using TF-IDF fallback.")

        # ── TF-IDF fallback ──────────────────────────────────────────────────
        print(f"[{self.name}] Using TF-IDF fallback.")
        self._index_ok = False
        self.collection = None
        self._build_tfidf(courses)

    def _build_tfidf(self, courses: list[Course]):
        all_tokens = [tokenize(c.full_text()) for c in courses]
        vocab: dict[str, int] = {}
        for toks in all_tokens:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        N = len(courses)

        def vec(toks: list[str]) -> list[float]:
            tf: dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            df = {t: sum(1 for ts in all_tokens if t in ts) for t in tf}
            v = [0.0] * len(vocab)
            for t, cnt in tf.items():
                if t in vocab:
                    v[vocab[t]] = (cnt / len(toks)) * (
                        math.log((N + 1) / (df[t] + 1)) + 1
                    )
            return v

        self._vocab = vocab
        self._vecs = {c.id: vec(all_tokens[i]) for i, c in enumerate(courses)}
        self._all_tokens = all_tokens

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)

    def process(self, profile: UserProfile, top_k: int = 3) -> list[RetrievalResult]:
        query = profile.search_query

        if self._index_ok and self.collection is not None:
            q_emb = self._embed(query)
            res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
            cmap = {c.id: c for c in self.courses}
            results = [
                RetrievalResult(cmap[cid], round(1.0 - dist, 4), "vector")
                for cid, dist in zip(res["ids"][0], res["distances"][0])
            ]
        else:
            q_toks = tokenize(query)
            df = {t: sum(1 for ts in self._all_tokens if t in ts) for t in q_toks}
            N = len(self.courses)
            q_vec = [0.0] * len(self._vocab)
            for t in q_toks:
                if t in self._vocab:
                    q_vec[self._vocab[t]] = (1.0 / max(len(q_toks), 1)) * (
                        math.log((N + 1) / (df.get(t, 0) + 1)) + 1
                    )
            results = sorted(
                [
                    RetrievalResult(
                        c, round(self._cosine(q_vec, self._vecs[c.id]), 4), "vector"
                    )
                    for c in self.courses
                ],
                key=lambda r: r.score,
                reverse=True,
            )[:top_k]

        print(
            f"[{self.name}] Top-{top_k}: "
            + ", ".join(f"{r.course.id}({r.score:.3f})" for r in results)
        )
        return results

