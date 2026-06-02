# ─────────────────────────────────────────────────────────────────────────────
#  Agent 1 — IntakeAgent  (function calling via Groq or Gemini)
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import Optional

from config.main import GEMINI_DEFAULT_MODEL, GROQ_DEFAULT_MODEL
from function.main import call_groq_with_tools, call_gemini_with_tools
from keywords.CourseKeywords import COURSE_KEYWORDS
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile, degree_from_year

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
                    "description": (
                        "Student's current year. Use this mapping:\n"
                        "  Undergraduate : 1 (freshman), 2 (sophomore), 3 (junior), 4 (senior)\n"
                        "  Master's      : 5 (1st year), 6 (2nd year)\n"
                        "  PhD           : 7 (1st year), 8 (2nd year), 9 (3rd year), 10 (4th year+)\n"
                        "Infer from context:\n"
                        "  'freshman' / 'new student' / no hint → 1\n"
                        "  'Master's' / 'grad student' / 'MSc'  → 5\n"
                        "  'PhD' / 'doctoral' / 'dissertation'  → 7\n"
                        "  '2nd year Master's'                  → 6\n"
                        "  '3rd year PhD'                       → 9"
                    ),
                    "minimum": 1,
                    "maximum": 10,
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

        try:
            return call_groq_with_tools(messages, [INTAKE_TOOL], model=self.model)
        except Exception as exc:
            print(f"[{self.name}] Groq failed: {exc}. Trying Gemini fallback.")
            return call_gemini_with_tools(messages, [INTAKE_TOOL], model=GEMINI_DEFAULT_MODEL)

    def _heuristic_fallback(self, raw_input: str) -> UserProfile:
        text = raw_input.lower()

        if re.search(r"\b(phd|doctoral|dissertation|candidacy)\b", text):
            year = 7
        elif re.search(r"\b(master|masters|msc|grad(uate)? student|thesis)\b", text):
            year = 5
        elif re.search(r"\b(first[- ]?year|1st[- ]?year|freshman|undergraduate|undergrad)\b", text):
            year = 1
        else:
            year = 1

        completed_courses = self._extract_completed_courses(raw_input)
        goals = self._extract_goals(raw_input)
        args = self._apply_language_hint(
            raw_input,
            {
                "academic_year": year,
                "completed_courses": completed_courses,
                "goals": goals,
                "constraints": [],
                "search_query": raw_input,
            },
        )

        return UserProfile(
            raw_input=raw_input,
            academic_year=year,
            degree_level=degree_from_year(year),  # tự map
            completed_courses=completed_courses,
            goals=goals,
            constraints=args.get("constraints") or [],
            search_query=raw_input,
            preferred_language=UserProfile._extract_preferred_language(args.get("constraints") or []),
            language_priority=UserProfile._extract_language_priority(args.get("constraints") or []),
        )

    @staticmethod
    def _extract_completed_courses(raw_input: str) -> list[str]:
        text = raw_input.lower()
        completion_intent = re.search(
            r"\b(completed|finished|passed|took|taken|done with|already did|have done|have finished)\b",
            text,
        )
        if not completion_intent:
            return []

        completed = []
        upper_input = raw_input.upper()
        for course_id in sorted(VALID_COURSE_IDS):
            if re.search(rf"\b{re.escape(course_id)}\b", upper_input):
                completed.append(course_id)

        for course in RAW_COURSES:
            if course["name"].lower() in text and course["id"] not in completed:
                completed.append(course["id"])

        return completed

    @staticmethod
    def _extract_goals(raw_input: str) -> list[str]:
        match = re.search(
            r"\b(?:want to|would like to|interested in|learn|study|looking for)\b\s+(?P<goal>.+)",
            raw_input,
            flags=re.IGNORECASE,
        )
        if not match:
            return []

        goal = re.sub(r"\b(course|class|recommendation)\b", "", match.group("goal"), flags=re.IGNORECASE)
        goal = re.sub(r"\s+", " ", goal).strip(" .")
        return [goal] if goal else []

    @staticmethod
    def _is_on_topic(text: str) -> bool:
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        profile_keywords = {
            "academic",
            "completed",
            "courses",
            "freshman",
            "first",
            "first-year",
            "firstyear",
            "profile",
            "student",
            "undergrad",
            "undergraduate",
            "update",
            "year",
        }
        if words & profile_keywords:
            return True

        generic_question_words = {
            "what", "which", "how", "when", "where", "why", "can", "should",
            "help", "assist", "information", "info", "details", "about",
            "regarding", "concerning",
        }
        topic_keywords = COURSE_KEYWORDS - generic_question_words
        return any(kw in words for kw in topic_keywords)

    @staticmethod
    def _apply_language_hint(raw_input: str, args: dict) -> dict:
        text = raw_input.lower()
        asks_language_filtered_course = any(
            phrase in text
            for phrase in (
                "prefer english",
                "preferred english",
                "english preferred",
                "english taught",
                "english-taught",
                "taught by english",
                "taught in english",
                "english course",
                "english courses",
                "prefer chinese",
                "preferred chinese",
                "chinese preferred",
                "chinese taught",
                "chinese-taught",
                "taught by chinese",
                "taught in chinese",
                "chinese course",
                "chinese courses",
            )
        )
        asks_about_two_languages = "english" in text and "chinese" in text
        if not asks_language_filtered_course or asks_about_two_languages:
            return args

        language = "English" if "english" in text else "Chinese"
        priority = "preferred" if re.search(r"\b(prefer|preferred|preference|if possible)\b", text) else "required"
        constraint = (
            f"{language} courses preferred"
            if priority == "preferred"
            else f"{language} courses only"
        )
        constraints = list(args.get("constraints") or [])
        if not any(constraint.lower() == existing.lower() for existing in constraints):
            constraints.append(constraint)
        args["constraints"] = constraints
        return args

    def process(
        self,
        raw_input: str,
        model: str | None = None,          # kept for backward compat, ignored if set in __init__
        existing_profile: Optional[UserProfile] = None,
    ) -> UserProfile | None:
        # reject off-topic input unless it includes course-related intent
        if not self._is_on_topic(raw_input):
            print(f"\n[{self.name}] Off-topic input rejected.")
            return None

        provider_label = self.provider.upper()
        mode = "Updating" if existing_profile else "Extracting"
        print(f"\n[{self.name}] {mode} user profile via {provider_label} …")

        system = INTAKE_SYSTEM
        if existing_profile:
            system += f"""
                EXISTING PROFILE (source of truth):
                academic_year    : {existing_profile.academic_year}
                degree_level     : {existing_profile.degree_level}
                completed_courses: {existing_profile.completed_courses}
                goals            : {existing_profile.goals}
                constraints      : {existing_profile.constraints}

                RULES FOR THIS TURN — read carefully:

                1. academic_year + degree_level (always update together):
                - Default: return existing values as-is
                - EXCEPTION: if student explicitly corrects their level
                    (e.g. "I'm actually PhD", "I realized I'm a master student")
                    → update BOTH fields together:
                    undergrad → academic_year between 1-4
                    master    → academic_year between 5-6
                    phd       → academic_year between 7-10

                2. completed_courses:
                - Return ONLY courses mentioned in THIS message as newly completed
                - Return [] if student did not mention completing anything new
                - Do NOT repeat: {existing_profile.completed_courses}

                3. goals:
                - Return ONLY goals that are genuinely new and not already in: {existing_profile.goals}
                - Return [] if no new goals mentioned
                - "research papers", "publishing", "thesis" are valid academic goals

                4. constraints:
                - Return ONLY constraints stated or corrected in THIS message
                - If a new constraint overlaps an old one, return the NEW replacement only
                  (example: old "English only", new "Chinese preferred" => return ["Chinese courses preferred"])
                - Return [] if no new constraints mentioned

                5. search_query:
                - Always reflect the student's CURRENT question in this message
                - Include degree level context: PhD/master students need graduate-level courses
                """

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": raw_input},
        ]

        try:
            args = self._call_llm(messages)
            args = self._apply_language_hint(raw_input, args)

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
            fallback_profile = self._heuristic_fallback(raw_input)
            if existing_profile:
                existing_profile.update(raw_input, {
                    "academic_year": fallback_profile.academic_year,
                    "completed_courses": fallback_profile.completed_courses,
                    "goals": fallback_profile.goals,
                    "constraints": fallback_profile.constraints,
                    "search_query": fallback_profile.search_query,
                })
                return existing_profile
            return fallback_profile

    def _build_profile(self, raw_input: str, args: dict) -> UserProfile:

        year = max(1, min(10, int(args.get("academic_year", 1))))
        degree_level = args.get("degree_level")
        if degree_level in ("undergrad", "master", "phd"):
            lo, hi = DEGREE_YEAR_RANGES[degree_level]
            year = max(lo, min(hi, year))
        else:
            degree_level = degree_from_year(year)

        completed = [c for c in (args.get("completed_courses") or []) if c in VALID_COURSE_IDS]

        return UserProfile(
            raw_input=raw_input,
            academic_year=year,
            degree_level=degree_level,
            completed_courses=completed,
            goals=args.get("goals") or [],
            constraints=args.get("constraints") or [],
            preferred_language=UserProfile._extract_preferred_language(args.get("constraints") or []),
            language_priority=UserProfile._extract_language_priority(args.get("constraints") or []),
            search_query=args.get("search_query", raw_input).strip(),
        )
