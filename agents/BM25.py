# ─────────────────────────────────────────────────────────────────────────────
#  Agent 2 — BM25Agent
# ─────────────────────────────────────────────────────────────────────────────
import sys

from function.main import tokenize
from models.Course import Course
from models.RetrievalResult import RetrievalResult
from models.UserProfile import UserProfile


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("pip install rank-bm25")



class BM25Agent:
    """Keyword retrieval using BM25Okapi from rank_bm25."""

    name = "BM25Agent"

    def __init__(self, courses: list[Course]):
        self.courses = courses
        corpus = [tokenize(c.full_text()) for c in courses]
        self.bm25 = BM25Okapi(corpus)
        print(f"[{self.name}] Indexed {len(courses)} courses.")

    def process(self, profile: UserProfile, top_k: int = 3) -> list[RetrievalResult]:
        tokens = tokenize(profile.search_query)
        scores = self.bm25.get_scores(tokens)
        results = sorted(
            [
                RetrievalResult(self.courses[i], round(float(scores[i]), 4), "bm25")
                for i in range(len(self.courses))
            ],
            key=lambda r: r.score,
            reverse=True,
        )[:top_k]
        print(
            f"[{self.name}] Top-{top_k}: "
            + ", ".join(f"{r.course.id}({r.score:.3f})" for r in results)
        )
        return results

