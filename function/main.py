# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import os
import re

from dotenv import load_dotenv
from groq import Groq

from groq import Groq
from config.main import GROQ_DEFAULT_MODEL
from config.main import GROQ_DEFAULT_MODEL

from models.Course import Course
from models.RetrievalResult import RetrievalResult
from difflib import SequenceMatcher


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    list_a: list[RetrievalResult],
    list_b: list[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    rrf: dict[str, float] = {}
    course_map: dict[str, Course] = {}
    for rank, res in enumerate(list_a, start=1):
        rrf[res.course.id] = rrf.get(res.course.id, 0.0) + 1.0 / (k + rank)
        course_map[res.course.id] = res.course
    for rank, res in enumerate(list_b, start=1):
        rrf[res.course.id] = rrf.get(res.course.id, 0.0) + 1.0 / (k + rank)
        course_map[res.course.id] = res.course
    return [
        RetrievalResult(course=course_map[cid], score=round(s, 6), source="fusion")
        for cid, s in sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    ]


def check_prerequisites_met(course: Course, completed_courses: list[str]) -> tuple[bool, list[str]]:
    completed_set = set(completed_courses)
    # Normalize: extract .id if prereqs are Course objects, otherwise use as-is
    prereq_ids = [
        p.id if hasattr(p, "id") else p
        for p in course.prerequisites
    ]
    missing = [p for p in prereq_ids if p not in completed_set]
    return len(missing) == 0, missing

# ─────────────────────────────────────────────────────────────────────────────
#  Groq client + shared tool-call helper
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_groq_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str = GROQ_DEFAULT_MODEL,
) -> dict:
    """
    Call the Groq API with tool definitions and return the first tool-call
    arguments as a parsed Python dict.

    Raises RuntimeError if the model returns a plain-text response instead
    of a tool call.
    """
    response = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        raise RuntimeError(
            f"Model did not invoke a tool. Content: {response_message.content}"
        )

    return json.loads(tool_calls[0].function.arguments)

def _is_similar_goal(self, new_goal: str, existing_goals: list[str], threshold: float = 0.75) -> bool:
    """Return True if new_goal is too similar to any existing goal."""
    new_lower = new_goal.lower()
    for g in existing_goals:
        ratio = SequenceMatcher(None, new_lower, g.lower()).ratio()
        if ratio >= threshold:
            return True
    return False