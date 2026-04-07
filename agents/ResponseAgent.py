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
                "",
                "  💡 SUGGESTIONS TO IMPROVE RECOMMENDATIONS",
                sep2,
                "  • Tell me about your academic interests or goals (e.g., 'I want to learn AI').",
                "  • Share courses you've already completed (e.g., 'I've finished CSIE1001 and MATH2001').",
                "  • Mention any scheduling constraints (e.g., 'Only Tuesdays and Thursdays').",
                "  • Specify your preferred language (e.g., 'English courses only').",
                "  • Ask about specific topics (e.g., 'What about machine learning?').",
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

        # ── Locked / Filtered Courses ─────────────────────────
        if locked_results:
            lines += ["", f"  🔒 LOCKED / NOT RECOMMENDED COURSES", sep2]
            for r in locked_results:
                lines.append(f"  ✗  [{r.course.id}] {r.course.name}")
                if r.filter_reason:
                    lines.append(f"       Reason: {r.filter_reason}")
                if r.missing_prereqs:
                    missing_names = [
                        f"{pid} ({course_map[pid].name})" if pid in course_map else pid
                        for pid in r.missing_prereqs
                    ]
                    lines.append(f"       Complete first: {', '.join(missing_names)}")

        lines += ["", sep, ""]
        return "\n".join(lines)

    def minimal_response(
        self,
        verdict: JudgeVerdict | None,
        course_map: dict[str, Course],
    ) -> str:
        if verdict and verdict.best_course_id in course_map:
            best = course_map[verdict.best_course_id]
            return f"I recommend [{best.id}] {best.name} because {verdict.reasoning}"
        return (
            "I couldn't find an eligible course based on your current profile. "
            "To help me give better recommendations, try telling me:\n"
            "• Your academic interests (e.g., 'I want to learn machine learning')\n"
            "• Courses you've completed (e.g., 'I've finished CSIE1001')\n"
            "• Scheduling preferences (e.g., 'Only mornings')\n"
            "• Language preferences (e.g., 'English courses')\n"
            "What would you like to share?"
        )
