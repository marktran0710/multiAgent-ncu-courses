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
import json
import google.genai as genai
import os
from google.genai import types


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

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def call_gemini_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str = "gemini-2.0-flash",
) -> dict:
    """
    Call Gemini with function calling tools using the new google-genai SDK.
    Returns function call arguments dict — same interface as call_groq_with_tools.
    """

    # ── extract system prompt ─────────────────────────────────────────
    system_instruction = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )

    # ── convert messages → Gemini Content format ──────────────────────
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            continue  # handled separately
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            )
        )

    # ── convert tool schemas (OpenAI format → Gemini format) ──────────
    gemini_functions = []
    for tool in tools:
        fn = tool["function"]
        gemini_functions.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=_convert_schema(fn["parameters"]),
            )
        )

    # ── config ────────────────────────────────────────────────────────
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[types.Tool(function_declarations=gemini_functions)],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",  # force a function call
            )
        ),
        temperature=0.2,
    )

    # ── call API ──────────────────────────────────────────────────────
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    # ── extract function call args ────────────────────────────────────
    for part in response.candidates[0].content.parts:
        if part.function_call:
            return dict(part.function_call.args)

    raise ValueError(
        f"[Gemini] No function call in response: {response.candidates[0].content}"
    )


# ── schema converter ──────────────────────────────────────────────────────────

def _convert_schema(schema: dict) -> types.Schema:
    """Recursively convert OpenAI-style JSON schema to Gemini types.Schema."""
    type_map = {
        "string":  "STRING",
        "integer": "INTEGER",
        "number":  "NUMBER",
        "boolean": "BOOLEAN",
        "array":   "ARRAY",
        "object":  "OBJECT",
    }

    prop_type = type_map.get(schema.get("type", "string"), "STRING")

    kwargs = {"type": prop_type}

    if "description" in schema:
        kwargs["description"] = schema["description"]

    if "enum" in schema:
        kwargs["enum"] = [str(e) for e in schema["enum"]]

    if prop_type == "ARRAY" and "items" in schema:
        kwargs["items"] = _convert_schema(schema["items"])

    if prop_type == "OBJECT" and "properties" in schema:
        kwargs["properties"] = {
            k: _convert_schema(v)
            for k, v in schema["properties"].items()
        }
        if "required" in schema:
            kwargs["required"] = schema["required"]

    return types.Schema(**kwargs)