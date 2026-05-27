import json
import os
import re

from config.main import GEMINI_DEFAULT_MODEL, GROQ_DEFAULT_MODEL
from function.main import call_gemini_with_tools, call_groq_with_tools
from models.UserProfile import UserProfile


EVALUATION_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_course_response",
        "description": (
            "Evaluate a course advising response before it is shown to the student. "
            "Check factual consistency with eligible/locked courses, prerequisite status, "
            "requested teaching language, and whether the wording is useful."
        ),
        "parameters": {
            "type": "object",
            "required": ["approved", "score", "issues", "revised_response"],
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": "True only if the response is safe and useful to show as-is.",
                },
                "score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Quality score from 1 poor to 5 excellent.",
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific issues found, or an empty list.",
                },
                "revised_response": {
                    "type": "string",
                    "description": (
                        "A corrected concise student-facing response. Return the original response "
                        "when approved. Never invent courses outside the provided eligible/locked data."
                    ),
                },
            },
        },
    },
}


EVALUATION_SYSTEM = """\
You are an independent academic QA reviewer for an NCU course advising chatbot.
Evaluate the draft response before the student sees it.

Rules:
- Use only the provided profile, eligible courses, locked courses, and verdict.
- Never approve a response that recommends a locked course as eligible.
- If no course is eligible but relevant locked courses exist, prefer pathway wording:
  "closest match ... not eligible yet ... complete first ..."
- Respect teaching language priority:
  required = non-matching language courses must not be recommended.
  preferred = matching language courses should be prioritized, but alternatives may remain.
- Keep revised_response concise, direct, and student-facing.
- Always call evaluate_course_response.
"""


class ResponseEvaluationAgent:
    name = "ResponseEvaluationAgent"

    def __init__(self, primary_provider: str = "groq", primary_model: str = GROQ_DEFAULT_MODEL):
        self.primary_provider = primary_provider
        self.primary_model = primary_model
        self.provider = "gemini" if primary_provider == "groq" else "groq"
        self.model = GEMINI_DEFAULT_MODEL if self.provider == "gemini" else GROQ_DEFAULT_MODEL

    def _provider_ready(self, provider: str) -> bool:
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
        key = os.environ.get(key_name, "").strip()
        return bool(key and key.lower() not in {"placeholder", "your_gemini_key_here", "your_groq_key_here"})

    def _evaluation_candidates(self) -> list[tuple[str, str]]:
        alternate_groq = (
            "llama-3.1-8b-instant"
            if self.primary_model != "llama-3.1-8b-instant"
            else "llama-3.3-70b-versatile"
        )
        if self.primary_provider == "groq":
            return [("gemini", GEMINI_DEFAULT_MODEL), ("groq", alternate_groq)]
        return [("groq", GROQ_DEFAULT_MODEL), ("gemini", GEMINI_DEFAULT_MODEL)]

    def _call_llm(self, messages: list[dict], provider: str, model: str) -> dict:
        if provider == "gemini":
            return call_gemini_with_tools(messages, [EVALUATION_TOOL], model=model)
        try:
            return call_groq_with_tools(messages, [EVALUATION_TOOL], model=model)
        except Exception as exc:
            parsed = self._parse_groq_failed_generation(str(exc))
            if parsed:
                return parsed
            raise

    @staticmethod
    def _parse_groq_failed_generation(error_text: str) -> dict | None:
        match = re.search(
            r"<function=evaluate_course_response>(.*?)</function>",
            error_text,
            flags=re.DOTALL,
        )
        if not match:
            return None
        payload = match.group(1).replace("\\\\'", "'").replace("\\'", "'")
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return {
            "approved": bool(parsed.get("approved")),
            "score": parsed.get("score", 3),
            "issues": parsed.get("issues") or [],
            "revised_response": parsed.get("revised_response") or "",
        }

    def process(
        self,
        draft_response: str,
        profile: UserProfile,
        eligible: list[dict],
        locked: list[dict],
        verdict: dict,
    ) -> tuple[str, dict]:
        payload = (
            f"== Student Profile ==\n{profile.describe()}\n\n"
            f"== Draft Response ==\n{draft_response}\n\n"
            f"== Verdict ==\n{verdict}\n\n"
            f"== Eligible Courses ==\n{eligible}\n\n"
            f"== Locked Courses ==\n{locked}"
        )
        messages = [
            {"role": "system", "content": EVALUATION_SYSTEM},
            {"role": "user", "content": payload},
        ]

        skipped = []
        for provider, model in self._evaluation_candidates():
            if not self._provider_ready(provider):
                skipped.append(f"{provider}:{model} not configured")
                continue
            try:
                result = self._call_llm(messages, provider, model)
            except Exception:
                skipped.append(f"{provider}:{model} failed")
                continue

            approved = bool(result.get("approved"))
            revised = str(result.get("revised_response") or "").strip()
            final_response = draft_response if approved or not revised else revised
            return final_response, {
                "provider": provider,
                "model": model,
                "status": "completed",
                "approved": approved,
                "score": result.get("score"),
                "issues": result.get("issues") or [],
                "attempted": skipped + [f"{provider}:{model} completed"],
            }

        return draft_response, {
            "provider": None,
            "model": None,
            "status": "skipped",
            "approved": True,
            "score": None,
            "issues": ["No secondary evaluation model was available."],
            "attempted": skipped,
        }
