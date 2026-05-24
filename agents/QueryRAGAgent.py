import math
from dataclasses import dataclass
from typing import Iterable

from function.main import tokenize
from models.Course import Course
from models.RetrievalResult import RetrievalResult
from models.UserProfile import UserProfile

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:
    raise RuntimeError("pip install rank-bm25") from exc


@dataclass(frozen=True)
class RetrievalStrategy:
    key: str
    label: str
    description: str


class QueryRAGAgent:
    """
    Query-time RAG retrieval strategies.

    The method names mirror common search-stack configurations:
      - Multi Match: terms can match across all course fields.
      - Best Fields: the strongest individual field match dominates.
      - KNN: dense semantic similarity when VectorAgent has embeddings.
      - Sparse Encoder: local sparse TF-IDF semantic approximation.
    """

    name = "QueryRAGAgent"

    STRATEGIES = [
        RetrievalStrategy("bm25_multi_match", "BM25 + Multi Match (MQ)", "keyword hybrid"),
        RetrievalStrategy("bm25_best_fields", "BM25 + Best Fields (BF)", "keyword + field aware"),
        RetrievalStrategy("knn_multi_match", "KNN + Multi Match", "dense semantic"),
        RetrievalStrategy("knn_best_fields", "KNN + Best Fields", "dense semantic + best field"),
        RetrievalStrategy("sparse_multi_match", "Sparse Encoder + Multi Match", "sparse semantic"),
        RetrievalStrategy("sparse_best_fields", "Sparse Encoder + Best Fields", "strongest method"),
    ]

    FIELD_WEIGHTS = {
        "id": 3.0,
        "name": 4.0,
        "department": 1.2,
        "prerequisites": 1.1,
        "language": 0.6,
        "degree": 0.6,
        "description": 2.4,
        "schedule": 0.4,
        "instructor": 0.3,
    }

    FUSION_WEIGHTS = {
        "bm25_agent": 1.00,
        "vector_agent": 1.10,
        "bm25_multi_match": 1.00,
        "bm25_best_fields": 1.10,
        "knn_multi_match": 1.15,
        "knn_best_fields": 1.20,
        "sparse_multi_match": 1.25,
        "sparse_best_fields": 1.35,
    }

    def __init__(self, courses: list[Course], dense_agent=None):
        self.courses = courses
        self.dense_agent = dense_agent
        self.course_map = {c.id: c for c in courses}
        self.fields_by_course = {c.id: self._course_fields(c) for c in courses}

        self.full_bm25 = BM25Okapi([self._weighted_tokens(self.fields_by_course[c.id]) for c in courses])
        self.field_bm25 = {
            field: BM25Okapi([tokenize(fields[field]) for fields in self.fields_by_course.values()])
            for field in self.FIELD_WEIGHTS
        }

        self._build_sparse_index()
        print(f"[{self.name}] Indexed {len(courses)} courses across {len(self.STRATEGIES)} strategies.")

    def process(self, profile: UserProfile, top_k: int = 6) -> dict[str, list[RetrievalResult]]:
        query = profile.search_query
        multi_scores = self._bm25_multi_match_scores(query)
        best_scores = self._bm25_best_fields_scores(query)
        dense_scores = self._dense_scores(profile)
        sparse_scores = self._sparse_scores(query)

        rankings = {
            "bm25_multi_match": self._rank(multi_scores, "bm25_multi_match", top_k),
            "bm25_best_fields": self._rank(best_scores, "bm25_best_fields", top_k),
            "knn_multi_match": self._rank(
                self._combine_scores(dense_scores, multi_scores, dense_weight=0.65),
                "knn_multi_match",
                top_k,
            ),
            "knn_best_fields": self._rank(
                self._combine_scores(dense_scores, best_scores, dense_weight=0.70),
                "knn_best_fields",
                top_k,
            ),
            "sparse_multi_match": self._rank(
                self._combine_scores(sparse_scores, multi_scores, dense_weight=0.70),
                "sparse_multi_match",
                top_k,
            ),
            "sparse_best_fields": self._rank(
                self._combine_scores(sparse_scores, best_scores, dense_weight=0.75),
                "sparse_best_fields",
                top_k,
            ),
        }

        summary = " | ".join(
            f"{strategy.label}: {self._ids(rankings[strategy.key])}"
            for strategy in self.STRATEGIES
        )
        print(f"[{self.name}] Top-{top_k} {summary}")
        return rankings

    def fuse(
        self,
        rankings: dict[str, list[RetrievalResult]],
        top_k: int | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        for source, results in rankings.items():
            weight = self.FUSION_WEIGHTS.get(source, 1.0)
            for rank, result in enumerate(results, start=1):
                scores[result.course.id] = scores.get(result.course.id, 0.0) + weight / (rrf_k + rank)

        fused = [
            RetrievalResult(self.course_map[cid], round(score, 6), "query_rag")
            for cid, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]
        return fused[:top_k] if top_k else fused

    def _course_fields(self, course: Course) -> dict[str, str]:
        return {
            "id": course.id,
            "name": course.name,
            "department": course.department,
            "prerequisites": " ".join(course.prerequisites) if course.prerequisites else "none",
            "language": course.language,
            "degree": course.degree,
            "description": course.description,
            "schedule": course.schedule,
            "instructor": course.instructor,
        }

    def _weighted_tokens(self, fields: dict[str, str]) -> list[str]:
        tokens: list[str] = []
        for field, text in fields.items():
            repeat = max(1, round(self.FIELD_WEIGHTS[field]))
            tokens.extend(tokenize(text) * repeat)
        return tokens

    def _bm25_multi_match_scores(self, query: str) -> dict[str, float]:
        query_tokens = tokenize(query)
        scores = self.full_bm25.get_scores(query_tokens)
        return {course.id: float(scores[i]) for i, course in enumerate(self.courses)}

    def _bm25_best_fields_scores(self, query: str, tie_breaker: float = 0.15) -> dict[str, float]:
        query_tokens = tokenize(query)
        scores_by_field = {
            field: bm25.get_scores(query_tokens)
            for field, bm25 in self.field_bm25.items()
        }

        scores: dict[str, float] = {}
        for i, course in enumerate(self.courses):
            weighted = [
                float(scores_by_field[field][i]) * self.FIELD_WEIGHTS[field]
                for field in self.FIELD_WEIGHTS
            ]
            if not weighted:
                scores[course.id] = 0.0
                continue
            best = max(weighted)
            scores[course.id] = best + tie_breaker * sum(s for s in weighted if s != best)
        return scores

    def _dense_scores(self, profile: UserProfile) -> dict[str, float]:
        if self.dense_agent is None:
            return {course.id: 0.0 for course in self.courses}

        try:
            dense_results = self.dense_agent.process(profile, top_k=len(self.courses))
        except Exception as exc:
            print(f"[{self.name}] WARNING: dense KNN unavailable ({exc}).")
            return {course.id: 0.0 for course in self.courses}

        scores = {course.id: 0.0 for course in self.courses}
        for result in dense_results:
            scores[result.course.id] = result.score
        return scores

    def _build_sparse_index(self) -> None:
        docs = [tokenize(course.full_text()) for course in self.courses]
        self._sparse_vocab = sorted({token for doc in docs for token in doc})
        self._sparse_index = {token: i for i, token in enumerate(self._sparse_vocab)}
        self._sparse_idf = self._idf(docs)
        self._sparse_vectors = {
            course.id: self._tfidf_vector(docs[i])
            for i, course in enumerate(self.courses)
        }

    def _sparse_scores(self, query: str) -> dict[str, float]:
        query_vector = self._tfidf_vector(tokenize(query))
        return {
            course.id: self._cosine(query_vector, self._sparse_vectors[course.id])
            for course in self.courses
        }

    def _idf(self, docs: list[list[str]]) -> dict[str, float]:
        total = len(docs)
        idf: dict[str, float] = {}
        for token in self._sparse_vocab:
            df = sum(1 for doc in docs if token in doc)
            idf[token] = math.log((total + 1) / (df + 1)) + 1
        return idf

    def _tfidf_vector(self, tokens: list[str]) -> list[float]:
        vector = [0.0] * len(self._sparse_vocab)
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            if token in self._sparse_index:
                counts[token] = counts.get(token, 0) + 1

        total = sum(counts.values()) or 1
        for token, count in counts.items():
            vector[self._sparse_index[token]] = (count / total) * self._sparse_idf[token]
        return vector

    def _combine_scores(
        self,
        semantic_scores: dict[str, float],
        lexical_scores: dict[str, float],
        dense_weight: float,
    ) -> dict[str, float]:
        semantic = self._normalize(semantic_scores)
        lexical = self._normalize(lexical_scores)
        lexical_weight = 1.0 - dense_weight
        return {
            course.id: dense_weight * semantic.get(course.id, 0.0)
            + lexical_weight * lexical.get(course.id, 0.0)
            for course in self.courses
        }

    def _normalize(self, scores: dict[str, float]) -> dict[str, float]:
        values = list(scores.values())
        if not values:
            return {}
        low, high = min(values), max(values)
        if math.isclose(low, high):
            return {key: 0.0 for key in scores}
        return {key: (value - low) / (high - low) for key, value in scores.items()}

    def _rank(
        self,
        scores: dict[str, float],
        source: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(self.course_map[cid], round(score, 4), source)
            for cid, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-9)

    @staticmethod
    def _ids(results: Iterable[RetrievalResult]) -> str:
        return ", ".join(result.course.id for result in list(results)[:3])
