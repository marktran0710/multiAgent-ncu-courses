# ─────────────────────────────────────────────────────────────────────────────
#  Agent 4 — FusionAgent  (RRF)
# ─────────────────────────────────────────────────────────────────────────────

from function.main import check_prerequisites_met, reciprocal_rank_fusion
from models.RetrievalResult import RetrievalResult
from models.UserProfile import UserProfile


class FusionAgent:
    """Merges BM25 and vector rankings via Reciprocal Rank Fusion (RRF)."""

    name = "FusionAgent"

    def process(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        profile: UserProfile,                          # <-- add this
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:   # <-- (eligible, locked)
        fused = reciprocal_rank_fusion(bm25_results, vector_results)

        eligible, locked = [], []
        for r in fused:
            met, missing = check_prerequisites_met(r.course, profile.completed_courses)
            r.missing_prereqs = missing          # attach for display
            if met:
                eligible.append(r)
            else:
                locked.append(r)

        print(
            f"[{self.name}] Eligible: {[r.course.id for r in eligible]} | "
            f"Locked: {[r.course.id for r in locked]}"
        )
        return eligible, locked