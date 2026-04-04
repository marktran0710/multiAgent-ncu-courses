# ─────────────────────────────────────────────────────────────────────────────
#  Agent 1 — IntakeAgent  (function calling via Groq or Gemini)
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import Optional

from config.main import GROQ_DEFAULT_MODEL
from function.main import call_groq_with_tools, call_gemini_with_tools
from keywords.CourseKeywords import COURSE_KEYWORDS
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile

INTAKE_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_user_profile",
        "description": (
            "Extract structured information about a student who is looking for course recommendations. "
            "Call this function once you have enough information from the student's message."
        ),
        "parameters": {
            "type": "object",
            "required": ["academic_year", "search_query"],
            "properties": {
                "academic_year": {
                    "type": "integer",
                    "description": "Student's current academic year (1=freshman, 2=sophomore, 3=junior, 4=senior). Infer from context if not stated; default to 1.",
                },
                "completed_courses": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(VALID_COURSE_IDS),
                    },
                    "description": (
                        "Course IDs the student has EXPLICITLY said they completed. "
                        "Use semantic meaning to identify the course — match based on topic, not exact wording:\n"
                        + "\n".join(f'  - {c["id"]}: {c["name"]} — {c["description"][:80]}' for c in RAW_COURSES)
                        + "\n\nRULES:\n"
                        "- Only include courses the student clearly stated they finished.\n"
                        "- Use course descriptions above to resolve ambiguous names semantically.\n"
                        "- If unsure which course the student means, do NOT guess — leave it out and set needs_clarification=true.\n"
                        "- Never include courses the student only mentioned wanting to take.\n"
                        "- 'discrete mathematics' → CSIE1002 (not MATH2001).\n"
                        "- 'linear algebra' → MATH2001 (not CSIE1002)."
                    ),
                },
                "needs_clarification": {
                    "type": "boolean",
                    "description": (
                        "Set to true if the student mentioned a course name that is ambiguous or "
                        "could not be confidently mapped to a course ID. "
                        "When true, the system will ask the student to clarify."
                    ),
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Only include if the student EXPLICITLY states what they want to learn or achieve. "
                        "Do NOT infer or hallucinate goals. Leave empty if not clearly stated."
                    ),
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any scheduling, prerequisite, or credit constraints mentioned by the student.",
                },
                "search_query": {
                    "type": "string",
                    "description": (
                        "A concise, information-rich search string (10–30 words) that captures "
                        "the student's learning goals. This will be used directly for BM25 and "
                        "vector retrieval. Make it specific and content-rich."
                    ),
                },
            },
        },
    },
}

INTAKE_SYSTEM = """\
You are an academic advisor assistant at NCU (National Central University), 
strictly limited to NCU CSIE course recommendations ONLY.

STRICT RULES:
- ONLY discuss NCU CSIE courses, prerequisites, schedules, and academic goals.
- If the student asks ANYTHING unrelated to courses or academics, refuse and redirect.
- Never answer questions about general knowledge, coding help, or off-topic subjects.
- Always call the extract_user_profile function — never reply with plain text.

IMPORTANT: Only populate 'goals' if the student explicitly mentions what they want 
to achieve. If they only mention completed courses or ask what to take next, 
leave goals empty — do not infer.

Be generous in inference: if the student says "I'm new to programming", infer
academic_year=1 and completed_courses=[].
"""


class IntakeAgent:
    name = "IntakeAgent"

    def __init__(self, model: str = GROQ_DEFAULT_MODEL, provider: str = "groq"):
        self.model    = model
        self.provider = provider

    def _call_llm(self, messages: list[dict]) -> dict:
        """Route to the correct LLM provider."""
        if self.provider == "gemini":
            return call_gemini_with_tools(messages, [INTAKE_TOOL], model=self.model)
        return call_groq_with_tools(messages, [INTAKE_TOOL], model=self.model)

    def _heuristic_fallback(self, raw_input: str) -> UserProfile:
        return UserProfile(
            raw_input=raw_input,
            academic_year=1,
            degree_level="undergrad",
            completed_courses=[],
            goals=[],
            constraints=[],
            search_query=raw_input,
        )

    @staticmethod
    def _is_on_topic(text: str) -> bool:
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        return any(kw in words for kw in COURSE_KEYWORDS)

    def process(
        self,
        raw_input: str,
        model: str | None = None,          # kept for backward compat, ignored if set in __init__
        existing_profile: Optional[UserProfile] = None,
    ) -> UserProfile | None:
        # skip off-topic guard on follow-up turns
        if existing_profile is None and not self._is_on_topic(raw_input):
            print(f"\n[{self.name}] Off-topic input rejected.")
            return None

        provider_label = self.provider.upper()
        mode = "Updating" if existing_profile else "Extracting"
        print(f"\n[{self.name}] {mode} user profile via {provider_label} …")

        system = INTAKE_SYSTEM
        if existing_profile:
            system += f"""

EXISTING PROFILE (source of truth — do not contradict):
academic_year    : {existing_profile.academic_year}  ← only increase
degree_level     : {existing_profile.degree_level}   ← only change if student says so
completed_courses: {existing_profile.completed_courses}  ← return ONLY new ones
goals            : {existing_profile.goals}  ← return ONLY new ones, empty list if none
constraints      : {existing_profile.constraints}

YOUR JOB THIS TURN:
- Return academic_year = {existing_profile.academic_year} unless student explicitly says otherwise
- Return completed_courses = [] unless student mentions completing something new
- Return goals = [] unless student mentions a genuinely new learning goal
- Return search_query that reflects the student's CURRENT question
"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": raw_input},
        ]

        try:
            args = self._call_llm(messages)

            if args.get("needs_clarification"):
                course_list = "\n".join(f"  {c['id']}: {c['name']}" for c in RAW_COURSES)
                print(
                    f"\n[{self.name}] Unsure about a course name. Please clarify:\n{course_list}"
                )
                return existing_profile

            if existing_profile:
                existing_profile.update(raw_input, args)
                return existing_profile
            return self._build_profile(raw_input, args)

        except Exception as exc:
            print(f"[{self.name}] LLM call failed: {exc}. Using heuristic fallback.")
            return existing_profile or self._heuristic_fallback(raw_input)

    def _build_profile(self, raw_input: str, args: dict) -> UserProfile:
        degree_level = args.get("degree_level", "undergrad")
        if degree_level not in DEGREE_YEAR_RANGES:
            degree_level = "undergrad"

        lo, hi = DEGREE_YEAR_RANGES[degree_level]
        academic_year = max(lo, min(hi, int(args.get("academic_year", lo))))
        completed     = [c for c in (args.get("completed_courses") or []) if c in VALID_COURSE_IDS]

        return UserProfile(
            raw_input=raw_input,
            academic_year=academic_year,
            degree_level=degree_level,
            completed_courses=completed,
            goals=args.get("goals") or [],
            constraints=args.get("constraints") or [],
            search_query=args.get("search_query", raw_input).strip(),
        )