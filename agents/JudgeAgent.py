# ─────────────────────────────────────────────────────────────────────────────
#  Agent 5 — JudgeAgent  (function calling via Groq)
# ─────────────────────────────────────────────────────────────────────────────

from config.main import GROQ_DEFAULT_MODEL
from function.main import call_gemini_with_tools, call_groq_with_tools
from models.JudgeVerdict import JudgeVerdict
from models.RetrievalResult import RetrievalResult
from models.UserProfile import UserProfile


JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "select_best_course",
        "description": (
            "After reviewing the student profile and the RRF-ranked candidate courses, "
            "select the single best course for this student and explain why."
        ),
        "parameters": {
            "type": "object",
            "required": ["best_course_id", "reasoning", "confidence"],
            "properties": {
                "best_course_id": {
                    "type": "string",
                    "description": "The course ID (e.g. CSIE1001) of the best match.",
                },
                "runner_up_id": {
                    "type": "string",
                    "description": "The course ID of the second-best option, if any.",
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "A 2–4 sentence explanation of why this course is the best fit "
                        "for this specific student, referencing their goals, year, and "
                        "completed courses."
                    ),
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": (
                        "Confidence in the recommendation: "
                        "high = clear best match, medium = reasonable match, low = uncertain."
                    ),
                },
            },
        },
    },
}

JUDGE_SYSTEM = """\
You are a senior academic advisor at NCU.
CRITICAL RULE: Never recommend a course that the student has already completed. Check the 'Completed' section of the Student Profile carefully.
Pick the SINGLE best course by:
  1. Prerequisites must be met (hard requirement)
  2. Schedule MUST match student constraints — if student says Tuesday/Thursday
     only, courses on other days must be REJECTED even if academically relevant
  3. Alignment with goals
  4. Appropriateness for academic year

Treat schedule constraints as a HARD filter, not a soft preference.
Always call select_best_course. Do not respond with plain text.
"""


class JudgeAgent:
    """
    Agent 5 — LLM-powered best-course selector (Groq).

    After RRF fusion produces a ranked list, calls Groq with the full student
    profile and all candidates, then uses the select_best_course function call
    to produce a single final verdict with structured reasoning.

    Falls back to returning the RRF #1 result if Groq is unavailable.
    """

    name = "JudgeAgent"

    def __init__(self, model: str = GROQ_DEFAULT_MODEL, provider: str = "groq"):
        self.model    = model
        self.provider = provider

    def _call_llm(self, messages: list[dict]) -> dict:
        if self.provider == "gemini":
            return call_gemini_with_tools(messages, [JUDGE_TOOL], model=self.model)
        return call_groq_with_tools(messages, [JUDGE_TOOL], model=self.model)
    
    def process(
        self,
        profile: UserProfile,
        fused_results: list[RetrievalResult],
    ) -> JudgeVerdict | None:
        if not fused_results:
            print(f"[{self.name}] No eligible courses — skipping judgment.")
            return None


        print(f"\n[{self.name}] Judging best course from {len(fused_results)} candidates …")

        candidates_text = "\n\n".join(
            f"Rank #{i+1} (RRF score: {r.score:.6f})\n{r.course.summary()}"
            for i, r in enumerate(fused_results)
        )

        user_msg = (
            f"== Student Profile ==\n{profile.describe()}\n\n"
            f"== RRF-Ranked Candidate Courses ==\n{candidates_text}"
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        try:
            args = self._call_llm(messages)
            return self._build_verdict(args, fused_results)
        except Exception as exc:
            print(f"[{self.name}] LLM call failed ({self.provider}): {exc}. Defaulting to RRF #1.")
            return self._fallback_verdict(fused_results)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_verdict(args: dict, fused: list[RetrievalResult]) -> JudgeVerdict:
        valid_ids = {r.course.id for r in fused}
        best_id = args.get("best_course_id", "")

        if best_id not in valid_ids:
            # Groq hallucinated — scan top 3 and pick the best fit by
            # checking if the course name appears in the reasoning text
            reasoning_lower = args.get("reasoning", "").lower()
            top3 = fused[:3]
            best_id = fused[0].course.id  # default to RRF #1
            for r in top3:
                if any(word in reasoning_lower for word in r.course.name.lower().split()):
                    best_id = r.course.id
                    break

        runner_up = args.get("runner_up_id")
        if runner_up not in valid_ids or runner_up == best_id:
            # assign runner-up as the next top-3 course that isn't best
            runner_up = next(
                (r.course.id for r in fused[:3] if r.course.id != best_id),
                None,
            )

        return JudgeVerdict(
            best_course_id=best_id,
            reasoning=args.get("reasoning", "No reasoning provided."),
            confidence=args.get("confidence", "medium"),
            runner_up_id=runner_up,
        )

    @staticmethod
    def _fallback_verdict(fused: list[RetrievalResult]) -> JudgeVerdict:
        best = fused[0]
        runner_up = fused[1].course.id if len(fused) > 1 else None
        return JudgeVerdict(
            best_course_id=best.course.id,
            reasoning=(
                f"Selected based on highest RRF score ({best.score:.6f}). "
                "Groq was unavailable for deeper reasoning."
            ),
            confidence="low",
            runner_up_id=runner_up,
        )
