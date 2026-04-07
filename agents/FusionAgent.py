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

        DEGREE_RANK = {
            "undergrad": 1,
            "master": 2,
            "phd": 3,
        }

        eligible, locked = [], []
        for r in fused:
            required_degree = r.course.degree.lower()
            profile_degree = profile.degree_level.lower()
            if DEGREE_RANK.get(required_degree, 1) > DEGREE_RANK.get(profile_degree, 1):
                r.filter_reason = (
                    f"Course requires {r.course.degree} level, but your profile is {profile.degree_level}."
                )
                r.missing_prereqs = []
                locked.append(r)
                continue

            preferred_language = profile.preferred_language
            if preferred_language and (not r.course.language or r.course.language.lower() != preferred_language.lower()):
                r.filter_reason = (
                    f"Course is offered in {r.course.language or 'English'}, but you requested {preferred_language}."
                )
                r.missing_prereqs = []
                locked.append(r)
                continue

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