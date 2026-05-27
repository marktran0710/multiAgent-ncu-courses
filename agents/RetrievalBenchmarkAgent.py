from dataclasses import dataclass

from agents.QueryRAGAgent import QueryRAGAgent
from models.RetrievalResult import RetrievalResult
from models.UserProfile import UserProfile


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    expected_ids: list[str]
    focus: str


class RetrievalBenchmarkAgent:
    """Deterministic retrieval benchmark for the project demo/admin view."""

    CASES = [
        BenchmarkCase(
            "ai_english",
            "English-taught AI and edge intelligence course",
            ["CSIE4010", "CSIE4005", "CSIE4002"],
            "English AI recommendation",
        ),
        BenchmarkCase(
            "machine_learning",
            "machine learning foundations with algorithms and linear algebra",
            ["CSIE4001", "CSIE5003"],
            "Machine learning",
        ),
        BenchmarkCase(
            "nlp",
            "natural language processing transformers multilingual retrieval augmented generation",
            ["CSIE4003", "CSIE5005"],
            "NLP and RAG",
        ),
        BenchmarkCase(
            "cloud",
            "cloud computing kubernetes distributed deployment observability",
            ["CSIE3007", "CSIE5006", "CSIE4007"],
            "Cloud systems",
        ),
        BenchmarkCase(
            "security",
            "security cryptography secure systems network defense",
            ["CSIE3004", "CSIE5004", "COMM3002"],
            "Security",
        ),
        BenchmarkCase(
            "data",
            "data mining big data analytics clustering evaluation metrics",
            ["CSIE3011", "CSIE4006"],
            "Data science",
        ),
        BenchmarkCase(
            "wireless",
            "wireless communications 5G mobile networks resource allocation",
            ["COMM3001", "COMM4001", "COMM5001"],
            "Wireless networks",
        ),
        BenchmarkCase(
            "circuits",
            "integrated circuit VLSI mixed signal semiconductor design",
            ["EE3003", "EE4001", "EE5003", "EE5001"],
            "Electrical engineering",
        ),
    ]

    METHOD_LABELS = {
        "bm25_agent": "BM25Agent (keyword)",
        "vector_agent": "VectorAgent (semantic)",
        "legacy_bm25_vector_rag": "Past BM25 + Vector FusionAgent",
        "bm25_multi_match": "BM25 + Multi Match (MQ)",
        "bm25_best_fields": "BM25 + Best Fields (BF)",
        "knn_multi_match": "KNN + Multi Match",
        "knn_best_fields": "KNN + Best Fields",
        "sparse_multi_match": "Sparse Encoder + Multi Match",
        "sparse_best_fields": "Sparse Encoder + Best Fields",
        "final_query_rag_fusion": "Final weighted RRF fusion",
    }
    LEGACY_METHODS = ["bm25_agent", "vector_agent"]
    LEGACY_FUSION_METHOD = "legacy_bm25_vector_rag"
    CURRENT_METHOD = "final_query_rag_fusion"

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def run(self, top_k: int = 10) -> dict:
        method_rows: dict[str, dict] = {}
        case_rows = []

        for case in self.CASES:
            profile = self._profile_for(case)
            bm25_results = self.orchestrator.bm25_agent.process(profile, top_k=top_k)
            vector_results = self.orchestrator.vector_agent.process(profile, top_k=top_k)
            query_rankings = self.orchestrator.query_rag_agent.process(profile, top_k=top_k)
            all_rankings = {
                "bm25_agent": bm25_results,
                "vector_agent": vector_results,
                **query_rankings,
            }
            legacy_fused = self.orchestrator.fusion_agent.fuse_rankings(
                bm25_results,
                vector_results,
            )[:top_k]
            for result in legacy_fused:
                result.source = self.LEGACY_FUSION_METHOD
            fused = self.orchestrator.query_rag_agent.fuse(all_rankings, top_k=top_k)
            all_rankings[self.LEGACY_FUSION_METHOD] = legacy_fused
            all_rankings["final_query_rag_fusion"] = fused

            case_results = []
            for method, results in all_rankings.items():
                metric = self._score_case(results, case.expected_ids)
                metric.update({
                    "method": method,
                    "label": self.METHOD_LABELS.get(method, method),
                    "top_courses": [r.course.id for r in results[:5]],
                })
                case_results.append(metric)
                row = method_rows.setdefault(
                    method,
                    {
                        "method": method,
                        "label": self.METHOD_LABELS.get(method, method),
                        "cases": 0,
                        "top1_hits": 0,
                        "recall_at_3_hits": 0,
                        "recall_at_5_hits": 0,
                        "mrr_total": 0.0,
                    },
                )
                row["cases"] += 1
                row["top1_hits"] += metric["top1"]
                row["recall_at_3_hits"] += metric["recall_at_3"]
                row["recall_at_5_hits"] += metric["recall_at_5"]
                row["mrr_total"] += metric["reciprocal_rank"]

            case_rows.append({
                "id": case.id,
                "focus": case.focus,
                "query": case.query,
                "expected_ids": case.expected_ids,
                "comparison": self._compare_case(case_results),
                "results": sorted(
                    case_results,
                    key=lambda item: (item["rank"] is None, item["rank"] or 999, item["method"]),
                ),
            })

        summary = [self._summarize_method(row) for row in method_rows.values()]
        summary.sort(
            key=lambda row: (
                row["recall_at_3"],
                row["mrr"],
                row["top1_accuracy"],
            ),
            reverse=True,
        )

        return {
            "case_count": len(self.CASES),
            "top_k": top_k,
            "metrics": {
                "top1_accuracy": "Expected course ranked first.",
                "recall_at_3": "Any expected course appears in the top 3.",
                "recall_at_5": "Any expected course appears in the top 5.",
                "mrr": "Mean reciprocal rank of the first expected course.",
            },
            "comparison": self._build_comparison(summary, case_rows),
            "summary": summary,
            "cases": case_rows,
        }

    def _profile_for(self, case: BenchmarkCase) -> UserProfile:
        return UserProfile(
            raw_input=case.query,
            academic_year=4,
            degree_level="undergrad",
            completed_courses=[],
            goals=[case.focus],
            constraints=[],
            search_query=case.query,
        )

    @staticmethod
    def _score_case(results: list[RetrievalResult], expected_ids: list[str]) -> dict:
        ranked_ids = [r.course.id for r in results]
        rank = next(
            (index + 1 for index, course_id in enumerate(ranked_ids) if course_id in expected_ids),
            None,
        )
        return {
            "rank": rank,
            "top1": int(rank == 1),
            "recall_at_3": int(rank is not None and rank <= 3),
            "recall_at_5": int(rank is not None and rank <= 5),
            "reciprocal_rank": round(1 / rank, 4) if rank else 0.0,
        }

    @staticmethod
    def _summarize_method(row: dict) -> dict:
        cases = row["cases"] or 1
        return {
            "method": row["method"],
            "label": row["label"],
            "cases": row["cases"],
            "top1_accuracy": round(row["top1_hits"] / cases, 3),
            "recall_at_3": round(row["recall_at_3_hits"] / cases, 3),
            "recall_at_5": round(row["recall_at_5_hits"] / cases, 3),
            "mrr": round(row["mrr_total"] / cases, 3),
        }

    def _compare_case(self, case_results: list[dict]) -> dict:
        by_method = {result["method"]: result for result in case_results}
        legacy_best = by_method[self.LEGACY_FUSION_METHOD]
        current = by_method[self.CURRENT_METHOD]
        legacy_rank = legacy_best["rank"]
        current_rank = current["rank"]

        if legacy_rank is None and current_rank is None:
            delta = 0
            outcome = "same"
        elif legacy_rank is None:
            delta = 999 - current_rank
            outcome = "improved"
        elif current_rank is None:
            delta = legacy_rank - 999
            outcome = "regressed"
        else:
            delta = legacy_rank - current_rank
            outcome = "improved" if delta > 0 else "regressed" if delta < 0 else "same"

        return {
            "legacy_method": legacy_best["method"],
            "legacy_label": legacy_best["label"],
            "legacy_rank": legacy_rank,
            "legacy_top_courses": legacy_best["top_courses"],
            "current_method": current["method"],
            "current_label": current["label"],
            "current_rank": current_rank,
            "current_top_courses": current["top_courses"],
            "rank_delta": delta,
            "outcome": outcome,
        }

    def _build_comparison(self, summary: list[dict], case_rows: list[dict]) -> dict:
        by_method = {row["method"]: row for row in summary}
        legacy_rows = [by_method[method] for method in self.LEGACY_METHODS if method in by_method]
        current = by_method[self.CURRENT_METHOD]
        legacy_best = max(
            legacy_rows,
            key=lambda row: (row["recall_at_3"], row["mrr"], row["top1_accuracy"]),
        )
        improvements = [
            case for case in case_rows
            if case["comparison"]["outcome"] == "improved"
        ]
        regressions = [
            case for case in case_rows
            if case["comparison"]["outcome"] == "regressed"
        ]
        same = [
            case for case in case_rows
            if case["comparison"]["outcome"] == "same"
        ]

        return {
            "title": "Past version vs current Query RAG",
            "baseline_note": "Past version used BM25Agent + VectorAgent through FusionAgent only.",
            "current_note": "Current version adds six Query RAG methods and weighted RRF fusion.",
            "legacy_methods": legacy_rows,
            "legacy_best": by_method[self.LEGACY_FUSION_METHOD],
            "legacy_components": legacy_rows,
            "current": current,
            "delta": {
                "top1_accuracy": round(current["top1_accuracy"] - by_method[self.LEGACY_FUSION_METHOD]["top1_accuracy"], 3),
                "recall_at_3": round(current["recall_at_3"] - by_method[self.LEGACY_FUSION_METHOD]["recall_at_3"], 3),
                "recall_at_5": round(current["recall_at_5"] - by_method[self.LEGACY_FUSION_METHOD]["recall_at_5"], 3),
                "mrr": round(current["mrr"] - by_method[self.LEGACY_FUSION_METHOD]["mrr"], 3),
            },
            "percentage_point_change": self._percentage_point_change(
                by_method[self.LEGACY_FUSION_METHOD],
                current,
            ),
            "relative_percent_change": self._relative_percent_change(
                by_method[self.LEGACY_FUSION_METHOD],
                current,
            ),
            "case_counts": {
                "improved": len(improvements),
                "same": len(same),
                "regressed": len(regressions),
            },
            "special_cases": improvements + regressions + same,
        }

    @staticmethod
    def _percentage_point_change(old: dict, new: dict) -> dict:
        return {
            metric: round((new[metric] - old[metric]) * 100, 1)
            for metric in ("top1_accuracy", "recall_at_3", "recall_at_5", "mrr")
        }

    @staticmethod
    def _relative_percent_change(old: dict, new: dict) -> dict:
        changes = {}
        for metric in ("top1_accuracy", "recall_at_3", "recall_at_5", "mrr"):
            old_value = old[metric]
            if old_value == 0:
                changes[metric] = None
            else:
                changes[metric] = round(((new[metric] - old_value) / old_value) * 100, 1)
        return changes
