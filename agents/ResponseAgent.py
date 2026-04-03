# ─────────────────────────────────────────────────────────────────────────────
#  Agent 6 — ResponseAgent
# ─────────────────────────────────────────────────────────────────────────────

from models.JudgeVerdict import JudgeVerdict
from models.UserProfile import UserProfile
from models.RetrievalResult import RetrievalResult
from models.Course import Course


class ResponseAgent:
    """Formats the final output combining all agent results."""

    name = "ResponseAgent"

    def process(
        self,
        profile: UserProfile,
        eligible_results: list[RetrievalResult],
        locked_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        verdict: JudgeVerdict | None,
        course_map: dict[str, Course],
    ) -> str:
        sep  = "═" * 65
        sep2 = "─" * 65

        lines = [
            f"\n{sep}",
            "  NCU Course Finder v2 — Personalized Recommendation",
            sep,
            "",
            "  STUDENT PROFILE",
            sep2,
            profile.describe(),
            "",
        ]

        # ── Best Recommendation ───────────────────────────────────────
        if verdict and verdict.best_course_id in course_map:
            best = course_map[verdict.best_course_id]
            runner_up = course_map.get(verdict.runner_up_id) if verdict.runner_up_id else None
            conf_stars = {"high": "★★★", "medium": "★★☆", "low": "★☆☆"}.get(verdict.confidence, "★☆☆")

            lines += [
                "  ✅  TOP RECOMMENDATION",
                sep2,
                f"  Course      : [{best.id}] {best.name}",
                f"  Instructor  : {best.instructor}",
                f"  Semester    : {best.semester}",
                f"  Schedule    : {best.schedule}",
                f"  Credits     : {best.credits}",
                f"  Prereqs     : {', '.join(best.prerequisites) or 'None'}",
                f"  Confidence  : {verdict.confidence.upper()}  {conf_stars}",
                "",
                "  Why this course?",
            ]
            # word-wrap reasoning
            words, buf = verdict.reasoning.split(), []
            for word in words:
                if sum(len(w) + 1 for w in buf) + len(word) > 62:
                    lines.append("    " + " ".join(buf))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                lines.append("    " + " ".join(buf))

            if runner_up:
                lines += ["", f"  🥈 Runner-up  : [{runner_up.id}] {runner_up.name}"]
        else:
            lines += [
                "  ⚠️  NO ELIGIBLE COURSES FOUND",
                sep2,
                "  You have not completed the prerequisites for any matched course.",
                "  Complete the courses listed below first, then try again.",
            ]

        # ── Eligible Courses (ranked) ─────────────────────────────────
        if eligible_results:
            lines += ["", f"  📋 ALL ELIGIBLE COURSES  ({len(eligible_results)} found)", sep2]
            bm25_rank = {r.course.id: i + 1 for i, r in enumerate(bm25_results)}
            vec_rank  = {r.course.id: i + 1 for i, r in enumerate(vector_results)}
            for rank, r in enumerate(eligible_results, 1):
                marker = "  ◄ recommended" if verdict and r.course.id == verdict.best_course_id else ""
                lines.append(
                    f"  #{rank}  [{r.course.id}] {r.course.name:<35} "
                    f"RRF={r.score:.5f}  BM25=#{bm25_rank.get(r.course.id,'—')}  "
                    f"Vec=#{vec_rank.get(r.course.id,'—')}{marker}"
                )

        # ── Locked Courses (prereqs not met) ─────────────────────────
        if locked_results:
            lines += ["", f"  🔒 LOCKED COURSES  (prerequisites not yet met)", sep2]
            for r in locked_results:
                missing_names = [
                    f"{pid} ({course_map[pid].name})" if pid in course_map else pid
                    for pid in r.missing_prereqs
                ]
                lines.append(f"  ✗  [{r.course.id}] {r.course.name}")
                lines.append(f"       Complete first: {', '.join(missing_names)}")

        lines += ["", sep, ""]
        return "\n".join(lines)
