from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeVerdict:
    best_course_id: str
    reasoning: str
    confidence: str  # "high" | "medium" | "low"
    runner_up_id: Optional[str] = None

