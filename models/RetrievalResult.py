from dataclasses import dataclass, field

from models.Course import Course

@dataclass
class RetrievalResult:
    course: Course
    score: float
    source: str  # "bm25" | "vector" | "fusion"
    missing_prereqs: list[str] = field(default_factory=list)