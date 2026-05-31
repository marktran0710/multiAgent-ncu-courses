import importlib.util

from function.main import tokenize
from models.Course import Course


RAGAS_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "course",
    "for",
    "from",
    "i",
    "in",
    "is",
    "learn",
    "of",
    "on",
    "or",
    "recommend",
    "study",
    "the",
    "to",
    "want",
    "with",
}


class RagasEvaluationAgent:
    """RAGAS-shaped benchmark evaluator with deterministic local fallback."""

    name = "RagasEvaluationAgent"

    METRIC_DESCRIPTIONS = {
        "context_precision": "Relevant course contexts should appear early in the retrieved list.",
        "context_recall": "Expected course contexts should be retrieved in the current top results.",
        "faithfulness": "The generated recommendation should be grounded in retrieved course contexts.",
        "answer_relevancy": "The recommended course context should match the user's query terms.",
        "answer_correctness": "The recommended course should match the benchmark expected course.",
        "overall": "Average of the RAGAS-style scores.",
    }

    def __init__(self, course_map: dict[str, Course]):
        self.course_map = course_map
        self.ragas_available = importlib.util.find_spec("ragas") is not None

    def evaluate(self, case_rows: list[dict]) -> dict:
        rows = [self._evaluate_case(row) for row in case_rows]
        summary = {
            metric: round(
                sum(row["scores"][metric] for row in rows) / (len(rows) or 1),
                3,
            )
            for metric in self.METRIC_DESCRIPTIONS
        }
        hard_cases = sorted(rows, key=lambda row: row["scores"]["overall"])[:8]

        return {
            "backend": (
                "ragas-compatible-local"
                if not self.ragas_available
                else "ragas-installed-local-adapter"
            ),
            "ragas_package_available": self.ragas_available,
            "note": (
                "Scores use the RAGAS metric shape with deterministic local scoring. "
                "Install and configure ragas with an evaluator LLM to replace this adapter with LLM-judged RAGAS."
            ),
            "case_count": len(rows),
            "metrics": self.METRIC_DESCRIPTIONS,
            "summary": summary,
            "hard_cases": hard_cases,
        }

    def _evaluate_case(self, case_row: dict) -> dict:
        comparison = case_row["comparison"]
        retrieved_ids = comparison.get("current_top_courses", [])
        expected_ids = case_row.get("expected_ids", [])
        recommended_id = retrieved_ids[0] if retrieved_ids else None

        scores = {
            "context_precision": self._context_precision(retrieved_ids, expected_ids),
            "context_recall": self._context_recall(retrieved_ids, expected_ids),
            "faithfulness": self._faithfulness(recommended_id, retrieved_ids),
            "answer_relevancy": self._answer_relevancy(
                case_row.get("query", ""),
                recommended_id,
            ),
            "answer_correctness": 1.0 if recommended_id in expected_ids else 0.0,
        }
        scores["overall"] = round(sum(scores.values()) / len(scores), 3)

        return {
            "id": case_row["id"],
            "query": case_row["query"],
            "focus": case_row["focus"],
            "expected_ids": expected_ids,
            "expected_courses": self._course_summaries(expected_ids),
            "recommended_id": recommended_id,
            "recommended_course": self._course_summary(recommended_id),
            "retrieved_ids": retrieved_ids,
            "retrieved_contexts": self._course_summaries(retrieved_ids),
            "why_hard": self._why_hard(expected_ids, recommended_id, retrieved_ids, scores),
            "scores": scores,
        }

    def _why_hard(
        self,
        expected_ids: list[str],
        recommended_id: str | None,
        retrieved_ids: list[str],
        scores: dict,
    ) -> str:
        expected = set(expected_ids)
        if recommended_id not in expected:
            expected_rank = next(
                (rank for rank, course_id in enumerate(retrieved_ids, start=1) if course_id in expected),
                None,
            )
            if expected_rank:
                return (
                    f"Expected course is retrieved at rank #{expected_rank}, "
                    f"but rank #1 is {recommended_id}."
                )
            return "Expected course is missing from the retrieved context list."
        if scores["context_precision"] < 1:
            return "Expected course is present, but not early enough in the retrieved context list."
        if scores["answer_relevancy"] < 1:
            return "Recommended course is correct, but query terms only partially overlap its context."
        return "This is one of the lowest-scoring remaining cases after sorting by overall RAGAS score."

    def _course_summaries(self, course_ids: list[str]) -> list[dict]:
        return [
            summary
            for summary in (self._course_summary(course_id) for course_id in course_ids)
            if summary
        ]

    def _course_summary(self, course_id: str | None) -> dict | None:
        course = self.course_map.get(course_id) if course_id else None
        if not course:
            return None

        return {
            "id": course.id,
            "name": course.name,
            "language": course.language,
            "degree": course.degree,
            "prerequisites": course.prerequisites,
            "description": self._shorten(course.description),
        }

    @staticmethod
    def _shorten(text: str, limit: int = 150) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _context_precision(retrieved_ids: list[str], expected_ids: list[str]) -> float:
        expected = set(expected_ids)
        if not expected:
            return 0.0

        hits = 0
        precisions = []
        for rank, course_id in enumerate(retrieved_ids, start=1):
            if course_id in expected:
                hits += 1
                precisions.append(hits / rank)

        return round(sum(precisions) / len(expected), 3) if precisions else 0.0

    @staticmethod
    def _context_recall(retrieved_ids: list[str], expected_ids: list[str]) -> float:
        expected = set(expected_ids)
        if not expected:
            return 0.0
        return round(len(expected & set(retrieved_ids)) / len(expected), 3)

    def _faithfulness(self, recommended_id: str | None, retrieved_ids: list[str]) -> float:
        if not recommended_id or recommended_id not in retrieved_ids:
            return 0.0

        recommended = self.course_map.get(recommended_id)
        if not recommended:
            return 0.0

        answer_terms = self._content_terms(
            f"{recommended.id} {recommended.name} {recommended.language} "
            f"{recommended.degree} {recommended.description}"
        )
        context_terms = self._content_terms(
            " ".join(
                self.course_map[course_id].full_text()
                for course_id in retrieved_ids
                if course_id in self.course_map
            )
        )
        if not answer_terms:
            return 0.0
        return round(len(answer_terms & context_terms) / len(answer_terms), 3)

    def _answer_relevancy(self, query: str, recommended_id: str | None) -> float:
        if not recommended_id or recommended_id not in self.course_map:
            return 0.0

        query_terms = self._content_terms(query)
        if not query_terms:
            return 0.0

        course_terms = self._content_terms(self.course_map[recommended_id].full_text())
        return round(len(query_terms & course_terms) / len(query_terms), 3)

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        return {
            token
            for token in tokenize(text)
            if len(token) > 2 and token not in RAGAS_STOP_WORDS
        }
