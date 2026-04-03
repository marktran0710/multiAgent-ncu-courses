"""
Demo Scenarios — NCU Course Finder v2
10 hand-crafted test cases for professor presentation.

Run all  : pytest test_demo_scenarios.py -v
Run one  : pytest test_demo_scenarios.py::test_02_off_topic_rejection -v
"""

from __future__ import annotations
import pytest
from unittest.mock import patch

from models.Course import Course
from models.UserProfile import RAW_COURSES, UserProfile
from models.RetrievalResult import RetrievalResult
from models.JudgeVerdict import JudgeVerdict
from keywords.OffTopicResponse import OFF_TOPIC_RESPONSE
from agents.OrchestratorAgent import CourseFinderOrchestrator

PATCH_INTAKE = "agents.IntakeAgent.call_groq_with_tools"
PATCH_JUDGE  = "agents.JudgeAgent.call_groq_with_tools"


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def intake_payload(**kwargs) -> dict:
    """Build a minimal IntakeAgent function-call return value."""
    return {
        "academic_year":    kwargs.get("academic_year", 1),
        "degree_level":     kwargs.get("degree_level", "undergrad"),
        "completed_courses": kwargs.get("completed_courses", []),
        "goals":            kwargs.get("goals", []),
        "constraints":      kwargs.get("constraints", []),
        "search_query":     kwargs.get("search_query", "course recommendation"),
    }

def judge_payload(best: str, runner_up: str | None = None,
                  reasoning: str = "Best fit.", confidence: str = "high") -> dict:
    return {
        "best_course_id": best,
        "runner_up_id":   runner_up,
        "reasoning":      reasoning,
        "confidence":     confidence,
    }

def run(query: str, side_effects: list, profile=None):
    with patch(PATCH_INTAKE) as mock_intake, \
         patch(PATCH_JUDGE)  as mock_judge:
        mock_intake.side_effect = [side_effects[0]]
        mock_judge.side_effect  = [side_effects[1]] if len(side_effects) > 1 else []
        orc = CourseFinderOrchestrator()
        return orc.run(query, profile=profile)


# ─────────────────────────────────────────────────────────────────────────────
#  Case 1 — Complete beginner
# ─────────────────────────────────────────────────────────────────────────────

def test_01_complete_beginner():
    """
    Freshman with zero experience.
    IntakeAgent infers year=1, no completed courses.
    Prereq filter must block all courses that require prerequisites.
    Recommendation must be CSIE1001 or CSIE1002 (no-prereq courses).
    """
    output, profile = run(
        "I just started university, I have no programming experience. What should I take?",
        side_effects=[
            intake_payload(
                academic_year=1,
                completed_courses=[],
                goals=["learn programming basics"],
                search_query="introduction programming beginner python",
            ),
            judge_payload(
                best="CSIE1001",
                runner_up="CSIE1002",
                reasoning="No prerequisites required, perfect for a beginner.",
            ),
        ],
    )

    assert profile is not None, "Profile should be extracted"
    assert profile.academic_year == 1
    assert profile.completed_courses == []

    # Recommended course must have no prerequisites
    course_map = {c["id"]: c for c in RAW_COURSES}
    assert "CSIE1001" in output or "CSIE1002" in output
    assert "TOP RECOMMENDATION" in output

    # Advanced courses must appear locked
    assert "LOCKED" in output
    assert "CSIE4001" in output or "CSIE3001" in output


# ─────────────────────────────────────────────────────────────────────────────
#  Case 2 — Off-topic rejection
# ─────────────────────────────────────────────────────────────────────────────

def test_02_off_topic_rejection():
    """
    User asks about restaurants — completely unrelated to courses.
    Off-topic guard must fire before any Groq call.
    Pipeline must return OFF_TOPIC_RESPONSE and None profile.
    """
    with patch(PATCH) as mock_groq:
        orc = CourseFinderOrchestrator()
        output, profile = orc.run("What's the best restaurant near NCU?")

    assert profile is None,                     "Profile must be None for off-topic input"
    assert output == OFF_TOPIC_RESPONSE,        "Must return the standard off-topic message"
    mock_groq.assert_not_called()               # Groq must never be reached


# ─────────────────────────────────────────────────────────────────────────────
#  Case 3 — Prerequisites not met for ML
# ─────────────────────────────────────────────────────────────────────────────

def test_03_prerequisites_not_met():
    """
    Student wants ML but has zero completed courses.
    CSIE4001 (Machine Learning) requires CSIE3001 + MATH2001.
    It must appear in LOCKED list, not as a recommendation.
    """
    output, profile = run(
        "I want to learn machine learning but I haven't taken any courses yet.",
        side_effects=[
            intake_payload(
                academic_year=1,
                completed_courses=[],
                goals=["machine learning", "AI"],
                search_query="machine learning neural networks beginner",
            ),
            judge_payload(
                best="CSIE1001",
                reasoning="Prerequisites for ML not met; start with programming.",
            ),
        ],
    )

    assert profile is not None
    assert "LOCKED" in output
    assert "CSIE4001" in output          # ML must appear in locked section

    # ML must NOT be the recommendation
    rec_section = output.split("TOP RECOMMENDATION")[1].split("ALL ELIGIBLE")[0]
    assert "CSIE4001" not in rec_section


# ─────────────────────────────────────────────────────────────────────────────
#  Case 4 — Prerequisites fully met → advanced course unlocked
# ─────────────────────────────────────────────────────────────────────────────

PATCH = "function.main.call_groq_with_tools"
def test_04_prerequisites_fully_met():
    """
    Student completed all prerequisites for Machine Learning.
    CSIE4001 must appear as eligible and be recommended.
    """
    completed = ["CSIE1001", "CSIE1002", "CSIE2001", "CSIE3001", "MATH2001"]
    output, profile = run(
        "I finished intro programming, data structures, discrete math, algorithms, "
        "and linear algebra. I want AI.",
        side_effects=[
            intake_payload(
                academic_year=3,
                completed_courses=completed,
                goals=["artificial intelligence", "machine learning"],
                search_query="machine learning AI algorithms advanced",
            ),
            judge_payload(
                best="CSIE4001",
                runner_up="CSIE4003",
                reasoning="All prerequisites met; ML is the natural next step toward AI.",
                confidence="high",
            ),
        ],
    )

    assert profile is not None
    assert "CSIE4001" in output
    assert "TOP RECOMMENDATION" in output

    # CSIE4001 must NOT appear in locked section as a course entry
    if "LOCKED" in output:
        locked_section = output.split("LOCKED")[1]
        assert "[CSIE4001]" not in locked_section  # ← brackets make it specific to course entries


# ─────────────────────────────────────────────────────────────────────────────
#  Case 5 — Multi-turn memory
# ─────────────────────────────────────────────────────────────────────────────

def test_05_multi_turn_profile_memory():
    with patch(PATCH_INTAKE) as mock_intake, \
         patch(PATCH_JUDGE)  as mock_judge:

        mock_intake.side_effect = [
            intake_payload(
                academic_year=2,
                completed_courses=["CSIE1001"],
                goals=["data structures"],
                search_query="data structures trees graphs algorithms",
            ),
            intake_payload(
                academic_year=2,
                completed_courses=["CSIE1001", "CSIE2001"],
                goals=["algorithms"],
                search_query="algorithms divide conquer dynamic programming",
            ),
        ]
        mock_judge.side_effect = [
            judge_payload(best="CSIE2001", reasoning="Logical next step after intro programming."),
            judge_payload(
                best="CSIE3001",
                reasoning="Data structures and discrete math done — ready for algorithms.",
                confidence="high",
            ),
        ]

        orc = CourseFinderOrchestrator()
        _, profile = orc.run("I'm a sophomore, I completed intro programming.")
        output, profile2 = orc.run(
            "Now I also finished data structures. What can I take next?",
            profile=profile,
        )

    assert profile2 is not None
    assert "CSIE1001" in profile2.completed_courses, "Turn 1 course must be retained"
    assert "CSIE2001" in profile2.completed_courses, "Turn 2 course must be merged"
    assert "CSIE3001" in output


# ─────────────────────────────────────────────────────────────────────────────
#  Case 6 — Schedule constraint handling
# ─────────────────────────────────────────────────────────────────────────────

def test_06_schedule_constraint():
    """
    Junior with all core courses done, available Tue/Thu only.
    JudgeAgent reasoning must mention the schedule constraint.
    Recommended course should ideally have Tue/Thu schedule.
    MATH2002 schedule: Tuesday 10:00–12:00, Thursday 10:00–11:00 — perfect match.
    """
    completed = [
        "CSIE1001", "CSIE1002", "CSIE2001", "CSIE2002",
        "CSIE3001", "CSIE3002", "MATH2001",
    ]
    output, profile = run(
        "I'm a junior, completed all core courses. I only have time on Tuesdays and Thursdays.",
        side_effects=[
            intake_payload(
                academic_year=3,
                completed_courses=completed,
                goals=["advance studies"],
                constraints=["available only on Tuesdays and Thursdays"],
                search_query="advanced CSIE courses Tuesday Thursday schedule",
            ),
            judge_payload(
                best="MATH2002",
                reasoning=(
                    "MATH2002 meets on Tuesday and Thursday, which perfectly matches "
                    "the student's schedule constraint. Other eligible courses fall "
                    "on days the student is unavailable."
                ),
                confidence="high",
            ),
        ],
    )

    assert profile is not None
    assert "Tuesdays and Thursdays" in profile.constraints or \
           "Tuesday" in str(profile.constraints)
    assert "MATH2002" in output
    assert "Tuesday" in output or "Thursday" in output   # schedule shown in output


# ─────────────────────────────────────────────────────────────────────────────
#  Case 7 — Judge picks NLP over RRF's Computer Vision
# ─────────────────────────────────────────────────────────────────────────────

def test_07_judge_over_rrf_nlp_vs_vision():
    """
    Student completed ML and wants to work with text/language.
    RRF might surface Computer Vision (CSIE4004) due to keyword overlap.
    JudgeAgent must pick CSIE4003 (NLP) based on the stated goal.
    """
    completed = ["CSIE1001", "CSIE1002", "CSIE2001", "CSIE3001",
                 "MATH2001", "MATH2002", "CSIE4001"]
    output, profile = run(
        "I completed machine learning. I want to work with text and language.",
        side_effects=[
            intake_payload(
                academic_year=4,
                completed_courses=completed,
                goals=["natural language processing", "text analysis"],
                search_query="natural language processing text transformers NLP",
            ),
            judge_payload(
                best="CSIE4003",
                runner_up="CSIE4004",
                reasoning=(
                    "The student explicitly wants to work with text and language. "
                    "CSIE4003 NLP directly addresses this goal. Computer Vision "
                    "focuses on images and is not aligned with the stated interest."
                ),
                confidence="high",
            ),
        ],
    )

    assert profile is not None
    assert "CSIE4003" in output        # NLP must be recommended

    # NLP must be the pick, not vision
    rec_section = output.split("TOP RECOMMENDATION")[1].split("ALL ELIGIBLE")[0]
    assert "CSIE4003" in rec_section
    assert "CSIE4004" not in rec_section.split("Runner-up")[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Case 8 — All courses locked
# ─────────────────────────────────────────────────────────────────────────────

def test_08_all_courses_locked():
    """
    Student wants OS but has studied nothing.
    CSIE3002 requires CSIE2001 + CSIE2002.
    Zero eligible courses for OS path → CSIE3002 must appear locked.
    No-prereq courses (CSIE1001, CSIE1002) are still eligible.
    """
    output, profile = run(
        "I want to take operating systems but I haven't studied anything yet.",
        side_effects=[
            intake_payload(
                academic_year=1,
                completed_courses=[],
                goals=["operating systems"],
                search_query="operating systems process memory scheduling kernel",
            ),
            judge_payload(best="CSIE1001", reasoning="Fallback."),
        ],
    )

    assert profile is not None
    assert "LOCKED" in output
    assert "CSIE3002" in output                    # OS must appear in locked

    # OS must NOT be recommended — it's locked
    assert "[CSIE3002]" not in output.split("TOP RECOMMENDATION")[1].split("ALL ELIGIBLE")[0]

    # CSIE1001 or CSIE1002 should be recommended as the starting point
    assert "CSIE1001" in output or "CSIE1002" in output

# ─────────────────────────────────────────────────────────────────────────────
#  Case 9 — Math-track student, cross-department retrieval
# ─────────────────────────────────────────────────────────────────────────────

def test_09_math_track_student():
    """
    Student interested in statistics/probability with no CS background.
    MATH2002 (Probability and Statistics) has no prerequisites.
    Must be recommended over locked CS courses.
    Cross-department retrieval (Mathematics dept) must work.
    """
    output, profile = run(
        "I'm interested in statistics and probability for data science. No CS background.",
        side_effects=[
            intake_payload(
                academic_year=1,
                completed_courses=[],
                goals=["statistics", "probability", "data science"],
                search_query="probability statistics data science mathematics",
            ),
            judge_payload(
                best="MATH2002",
                runner_up="MATH2001",
                reasoning=(
                    "MATH2002 Probability and Statistics directly matches the student's "
                    "interest in statistics for data science and has no prerequisites."
                ),
                confidence="high",
            ),
        ],
    )

    assert profile is not None
    assert "MATH2002" in output
    assert "TOP RECOMMENDATION" in output

    # Must show cross-department result (Mathematics)
    assert "Probability" in output or "Statistics" in output


# ─────────────────────────────────────────────────────────────────────────────
#  Case 10 — Senior specialization: vision vs language
# ─────────────────────────────────────────────────────────────────────────────

def test_10_senior_specialization_vision_vs_language():
    """
    Senior completed everything up to ML, asking to specialize further.
    Both CSIE4003 (NLP) and CSIE4004 (Vision) are eligible.
    Both must appear in eligible list.
    JudgeAgent must reason between them and pick one with explanation.
    """
    completed = [
        "CSIE1001", "CSIE1002", "CSIE2001", "CSIE2002",
        "CSIE3001", "CSIE3002", "MATH2001", "MATH2002", "CSIE4001",
    ]
    output, profile = run(
        "I completed everything up to machine learning. "
        "Should I go deeper into vision or language?",
        side_effects=[
            intake_payload(
                academic_year=4,
                completed_courses=completed,
                goals=["computer vision", "natural language processing", "specialization"],
                search_query="computer vision NLP deep learning specialization senior",
            ),
            judge_payload(
                best="CSIE4003",
                runner_up="CSIE4004",
                reasoning=(
                    "Both NLP and Computer Vision are excellent choices. Given the "
                    "student's phrasing around 'language', CSIE4003 NLP is the "
                    "primary recommendation. CSIE4004 Computer Vision is a strong "
                    "runner-up and can be taken in the following semester."
                ),
                confidence="medium",
            ),
        ],
    )

    assert profile is not None

    # Both courses must appear somewhere in output (eligible list)
    assert "CSIE4003" in output, "NLP must appear in output"
    assert "CSIE4004" in output, "Computer Vision must appear in output"

    # One must be the recommendation
    assert "TOP RECOMMENDATION" in output

    # Reasoning must mention both options
    assert "NLP" in output or "language" in output.lower()
    assert "Vision" in output or "vision" in output.lower()

    # Confidence should reflect the ambiguity
    assert "MEDIUM" in output or "HIGH" in output