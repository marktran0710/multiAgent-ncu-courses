"""
Tests for Multi-Agent Course Finder v2
Covers: utilities, agents, prerequisite logic, topic guard, pipeline
Run: pytest test_course_finder.py -v
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
#  Import everything from main (adjust path if needed)
# ─────────────────────────────────────────────────────────────────────────────
from models.Course import Course
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile
from models.RetrievalResult import RetrievalResult
from models.JudgeVerdict import JudgeVerdict
from keywords.CourseKeywords import COURSE_KEYWORDS
from keywords.OffTopicResponse import OFF_TOPIC_RESPONSE
from function.main import tokenize, reciprocal_rank_fusion, check_prerequisites_met, call_groq_with_tools
from fastapi.testclient import TestClient
import api as api_module
from api import app
from agents.IntakeAgent import IntakeAgent
from agents.BM25 import BM25Agent
from agents.VectorAgent import VectorAgent
from agents.FusionAgent import FusionAgent
from agents.JudgeAgent import JudgeAgent
from agents.ResponseAgent import ResponseAgent
from agents.OrchestratorAgent import CourseFinderOrchestrator
# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def all_courses() -> list[Course]:
    return [Course(**c) for c in RAW_COURSES]

@pytest.fixture
def course_map(all_courses) -> dict[str, Course]:
    return {c.id: c for c in all_courses}

@pytest.fixture
def fresh_profile() -> UserProfile:
    return UserProfile(
        raw_input="I want to learn machine learning",
        academic_year=3,
        degree_level="undergrad",
        completed_courses=["CSIE1001", "CSIE2001", "CSIE1002", "MATH2001"],
        goals=["machine learning", "AI"],
        constraints=[],
        search_query="machine learning neural networks supervised unsupervised",
    )

@pytest.fixture
def beginner_profile() -> UserProfile:
    return UserProfile(
        raw_input="I am a freshman with no experience",
        academic_year=1,
        degree_level="undergrad",
        completed_courses=[],
        goals=["learn programming basics"],
        constraints=[],
        search_query="introduction programming beginner python",
    )

@pytest.fixture
def bm25_agent(all_courses) -> BM25Agent:
    return BM25Agent(all_courses)

@pytest.fixture
def vector_agent(all_courses) -> VectorAgent:
    return VectorAgent(all_courses)

@pytest.fixture
def fusion_agent() -> FusionAgent:
    return FusionAgent()

@pytest.fixture
def response_agent() -> ResponseAgent:
    return ResponseAgent()


# ─────────────────────────────────────────────────────────────────────────────
#  1. Utility — tokenize
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercase(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenize("data-structures, trees!") == ["data", "structures", "trees"]

    def test_alphanumeric_kept(self):
        assert "csie1001" in tokenize("CSIE1001 course")

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_kept(self):
        assert "3" in tokenize("year 3 student")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Utility — check_prerequisites_met
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckPrerequisites:
    def test_no_prereqs_always_eligible(self, course_map):
        course = course_map["CSIE1001"]   # prerequisites = []
        met, missing = check_prerequisites_met(course, [])
        assert met is True
        assert missing == []

    def test_prereqs_met(self, course_map):
        course = course_map["CSIE2001"]   # requires CSIE1001
        met, missing = check_prerequisites_met(course, ["CSIE1001"])
        assert met is True
        assert missing == []

    def test_prereqs_not_met(self, course_map):
        course = course_map["CSIE2001"]   # requires CSIE1001
        met, missing = check_prerequisites_met(course, [])
        assert met is False
        assert "CSIE1001" in missing

    def test_partial_prereqs(self, course_map):
        course = course_map["CSIE3001"]   # requires CSIE2001 + CSIE1002
        met, missing = check_prerequisites_met(course, ["CSIE2001"])
        assert met is False
        assert "CSIE1002" in missing
        assert "CSIE2001" not in missing

    def test_all_prereqs_met_multi(self, course_map):
        course = course_map["CSIE3001"]   # requires CSIE2001 + CSIE1002
        met, missing = check_prerequisites_met(course, ["CSIE2001", "CSIE1002"])
        assert met is True
        assert missing == []

    def test_deep_chain(self, course_map):
        # CSIE4002 (Deep Learning) requires CSIE4001 (Machine Learning)
        # which itself requires CSIE3001 + MATH2001
        course = course_map["CSIE4002"]
        met, missing = check_prerequisites_met(course, [])
        assert met is False
        assert "CSIE4001" in missing


# ─────────────────────────────────────────────────────────────────────────────
#  3. Utility — reciprocal_rank_fusion
# ─────────────────────────────────────────────────────────────────────────────

class TestRRF:
    def _make_result(self, course_id: str, score: float, all_courses) -> RetrievalResult:
        course_map = {c.id: Course(**c) for c in RAW_COURSES}
        return RetrievalResult(course=course_map[course_id], score=score, source="test")

    def test_top_result_appears_in_both_lists_wins(self, all_courses):
        a = [
            RetrievalResult(Course(**RAW_COURSES[0]), 0.9, "bm25"),
            RetrievalResult(Course(**RAW_COURSES[1]), 0.5, "bm25"),
        ]
        b = [
            RetrievalResult(Course(**RAW_COURSES[0]), 0.8, "vector"),
            RetrievalResult(Course(**RAW_COURSES[2]), 0.4, "vector"),
        ]
        fused = reciprocal_rank_fusion(a, b)
        assert fused[0].course.id == RAW_COURSES[0]["id"]

    def test_output_source_is_fusion(self, all_courses):
        a = [RetrievalResult(Course(**RAW_COURSES[0]), 0.9, "bm25")]
        b = [RetrievalResult(Course(**RAW_COURSES[0]), 0.8, "vector")]
        fused = reciprocal_rank_fusion(a, b)
        assert all(r.source == "fusion" for r in fused)

    def test_no_duplicates_in_output(self, all_courses):
        course = Course(**RAW_COURSES[0])
        a = [RetrievalResult(course, 0.9, "bm25")]
        b = [RetrievalResult(course, 0.8, "vector")]
        fused = reciprocal_rank_fusion(a, b)
        ids = [r.course.id for r in fused]
        assert len(ids) == len(set(ids))

    def test_scores_are_positive(self, all_courses):
        a = [RetrievalResult(Course(**c), 0.5, "bm25") for c in RAW_COURSES[:3]]
        b = [RetrievalResult(Course(**c), 0.5, "vector") for c in RAW_COURSES[2:5]]
        fused = reciprocal_rank_fusion(a, b)
        assert all(r.score > 0 for r in fused)


# ─────────────────────────────────────────────────────────────────────────────
#  4. BM25Agent
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25Agent:
    def test_returns_top_k(self, bm25_agent, fresh_profile):
        results = bm25_agent.process(fresh_profile, top_k=3)
        assert len(results) == 3

    def test_scores_are_float(self, bm25_agent, fresh_profile):
        results = bm25_agent.process(fresh_profile)
        assert all(isinstance(r.score, float) for r in results)

    def test_source_is_bm25(self, bm25_agent, fresh_profile):
        results = bm25_agent.process(fresh_profile)
        assert all(r.source == "bm25" for r in results)

    def test_machine_learning_query_finds_ml_course(self, bm25_agent, fresh_profile):
        results = bm25_agent.process(fresh_profile, top_k=5)
        ids = [r.course.id for r in results]
        assert "CSIE4001" in ids   # Machine Learning course

    def test_programming_query_finds_intro_course(self, bm25_agent, beginner_profile):
        results = bm25_agent.process(beginner_profile, top_k=3)
        ids = [r.course.id for r in results]
        assert "CSIE1001" in ids   # Intro to Programming

    def test_results_sorted_descending(self, bm25_agent, fresh_profile):
        results = bm25_agent.process(fresh_profile, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_1(self, bm25_agent, fresh_profile):
        results = bm25_agent.process(fresh_profile, top_k=1)
        assert len(results) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  5. VectorAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorAgent:
    def test_returns_top_k(self, vector_agent, fresh_profile):
        results = vector_agent.process(fresh_profile, top_k=3)
        assert len(results) == 3

    def test_source_is_vector(self, vector_agent, fresh_profile):
        results = vector_agent.process(fresh_profile)
        assert all(r.source == "vector" for r in results)

    def test_results_sorted_descending(self, vector_agent, fresh_profile):
        results = vector_agent.process(fresh_profile, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_nlp_query_finds_nlp_course(self, vector_agent):
        profile = UserProfile(
            raw_input="I want NLP",
            academic_year=4,
            degree_level="undergrad",
            completed_courses=["CSIE4001", "MATH2002"],
            goals=["natural language processing"],
            constraints=[],
            search_query="natural language processing transformers text classification",
        )
        results = vector_agent.process(profile, top_k=5)
        ids = [r.course.id for r in results]
        assert "CSIE4003" in ids


# ─────────────────────────────────────────────────────────────────────────────
#  6. FusionAgent — prerequisite filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionAgent:
    def _make_results(self, ids: list[str], source: str) -> list[RetrievalResult]:
        course_map = {c["id"]: Course(**c) for c in RAW_COURSES}
        return [RetrievalResult(course_map[cid], 0.5, source) for cid in ids]

    def test_eligible_vs_locked_split(self, fusion_agent, beginner_profile):
        # Beginner has no completed courses — all courses with prereqs should be locked
        bm25 = self._make_results(["CSIE1001", "CSIE2001"], "bm25")
        vec  = self._make_results(["CSIE1001", "CSIE2001"], "vector")
        eligible, locked = fusion_agent.process(bm25, vec, beginner_profile)
        eligible_ids = [r.course.id for r in eligible]
        locked_ids   = [r.course.id for r in locked]
        assert "CSIE1001" in eligible_ids   # no prereqs
        assert "CSIE2001" in locked_ids     # requires CSIE1001

    def test_all_eligible_when_prereqs_met(self, fusion_agent, fresh_profile):
        # fresh_profile has CSIE1001, CSIE2001, CSIE1002, MATH2001 completed
        bm25 = self._make_results(["CSIE3001", "CSIE1001"], "bm25")
        vec  = self._make_results(["CSIE3001", "CSIE1001"], "vector")
        eligible, locked = fusion_agent.process(bm25, vec, fresh_profile)
        eligible_ids = [r.course.id for r in eligible]
        assert "CSIE3001" in eligible_ids
        assert locked == []

    def test_missing_prereqs_attached_to_locked(self, fusion_agent, beginner_profile):
        bm25 = self._make_results(["CSIE2001"], "bm25")
        vec  = self._make_results(["CSIE2001"], "vector")
        _, locked = fusion_agent.process(bm25, vec, beginner_profile)
        assert len(locked) == 1
        assert "CSIE1001" in locked[0].missing_prereqs

    def test_degree_level_constraints_lock_higher_level_courses(self, fusion_agent, beginner_profile):
        base_course = Course(**RAW_COURSES[0], degree="master")
        result = RetrievalResult(base_course, 0.5, "fusion")
        eligible, locked = fusion_agent.process([result], [result], beginner_profile)
        assert eligible == []
        assert len(locked) == 1
        assert "master" in locked[0].filter_reason.lower()

    def test_language_constraints_filter_non_matching_courses(self, fusion_agent):
        profile = UserProfile(
            raw_input="I need an English course",
            academic_year=3,
            degree_level="undergrad",
            completed_courses=["CSIE1001", "CSIE2001", "CSIE1002", "MATH2001"],
            goals=["natural language processing"],
            constraints=["English only"],
            preferred_language="English",
            search_query="natural language processing English",
        )
        course = Course(**RAW_COURSES[0], language="Chinese")
        result = RetrievalResult(course, 0.5, "fusion")
        eligible, locked = fusion_agent.process([result], [result], profile)
        assert eligible == []
        assert len(locked) == 1
        assert "english" in locked[0].filter_reason.lower()

    def test_empty_input(self, fusion_agent, fresh_profile):
        eligible, locked = fusion_agent.process([], [], fresh_profile)
        assert eligible == []
        assert locked == []


# ─────────────────────────────────────────────────────────────────────────────
#  7. IntakeAgent — topic guard
# ─────────────────────────────────────────────────────────────────────────────

class TestIntakeAgentTopicGuard:
    def test_off_topic_returns_none(self):
        agent = IntakeAgent()
        result = agent.process("What is the weather today?")
        assert result is None

    def test_off_topic_cooking_returns_none(self):
        agent = IntakeAgent()
        result = agent.process("How do I make pasta carbonara?")
        assert result is None

    def test_on_topic_course_passes_guard(self):
        agent = IntakeAgent()
        with patch("function.main.call_groq_with_tools") as mock_groq:
            mock_groq.return_value = {
                "academic_year": 1,
                "completed_courses": [],
                "goals": ["learn programming"],
                "constraints": [],
                "search_query": "introduction programming python",
            }
            result = agent.process("I want to learn programming")
        assert result is not None

    def test_on_topic_goal_passes_guard(self):
        agent = IntakeAgent()
        with patch("function.main.call_groq_with_tools") as mock_groq:
            mock_groq.return_value = {
                "academic_year": 3,
                "completed_courses": ["CSIE1001"],
                "goals": ["machine learning"],
                "constraints": [],
                "search_query": "machine learning AI neural networks",
            }
            result = agent.process("I want to study machine learning this semester")
        assert result is not None

    def test_off_topic_sports_returns_none(self):
        agent = IntakeAgent()
        result = agent.process("Who won the NBA finals?")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
#  8. IntakeAgent — profile building
# ─────────────────────────────────────────────────────────────────────────────

class TestIntakeAgentProfileBuilding:
    def _make_agent_with_mock(self, groq_return: dict) -> tuple[IntakeAgent, MagicMock]:
        agent = IntakeAgent()
        mock = MagicMock(return_value=groq_return)
        return agent, mock

    def test_build_profile_year_clamped(self):
        agent = IntakeAgent()
        args = {
            "academic_year": 99,   # out of range
            "degree_level": "undergrad",
            "completed_courses": [],
            "goals": ["learn"],
            "constraints": [],
            "search_query": "intro course",
        }
        profile = agent._build_profile("test", args)
        assert profile.academic_year <= DEGREE_YEAR_RANGES["undergrad"][1]

    def test_build_profile_invalid_course_ids_filtered(self):
        agent = IntakeAgent()
        args = {
            "academic_year": 1,
            "degree_level": "undergrad",
            "completed_courses": ["FAKE999", "CSIE1001"],
            "goals": [],
            "constraints": [],
            "search_query": "test",
        }
        profile = agent._build_profile("test", args)
        assert "FAKE999" not in profile.completed_courses
        assert "CSIE1001" in profile.completed_courses

    def test_heuristic_fallback_returns_valid_profile(self):
        agent = IntakeAgent()
        profile = agent._heuristic_fallback("I want algorithms")
        assert isinstance(profile, UserProfile)
        assert profile.academic_year == 1
        assert profile.degree_level == "undergrad"

    def test_update_merges_goals(self):
        agent = IntakeAgent()
        existing = UserProfile(
            raw_input="old",
            academic_year=2,
            degree_level="undergrad",
            completed_courses=["CSIE1001"],
            goals=["programming"],
            constraints=[],
            search_query="programming",
        )
        args = {
            "goals": ["machine learning"],
            "search_query": "machine learning",
        }
        existing.update("new input", args)
        assert "programming" in existing.goals
        assert "machine learning" in existing.goals

    def test_update_does_not_duplicate_completed_courses(self):
        agent = IntakeAgent()
        existing = UserProfile(
            raw_input="old",
            academic_year=2,
            degree_level="undergrad",
            completed_courses=["CSIE1001"],
            goals=[],
            constraints=[],
            search_query="test",
        )
        args = {"completed_courses": ["CSIE1001"]}   # already in list
        existing.update("new", args)
        assert existing.completed_courses.count("CSIE1001") == 1


# ─────────────────────────────────────────────────────────────────────────────
#  9. JudgeAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestJudgeAgent:
    def test_returns_none_when_no_eligible_courses(self, fresh_profile):
        agent = JudgeAgent()
        result = agent.process(fresh_profile, fused_results=[])
        assert result is None

    def test_fallback_verdict_uses_rrf_top(self, all_courses):
        agent = JudgeAgent()
        results = [
            RetrievalResult(all_courses[0], 0.9, "fusion"),
            RetrievalResult(all_courses[1], 0.5, "fusion"),
        ]
        verdict = agent._fallback_verdict(results)
        assert verdict.best_course_id == all_courses[0].id
        assert verdict.confidence == "low"

    def test_build_verdict_filters_hallucinated_id(self, all_courses):
        agent = JudgeAgent()
        results = [RetrievalResult(all_courses[0], 0.9, "fusion")]
        args = {
            "best_course_id": "FAKE999",   # hallucinated
            "reasoning": "some reason",
            "confidence": "high",
        }
        verdict = agent._build_verdict(args, results)
        assert verdict.best_course_id == all_courses[0].id  # falls back to #1

    def test_build_verdict_valid_id(self, all_courses):
        agent = JudgeAgent()
        results = [
            RetrievalResult(all_courses[0], 0.9, "fusion"),
            RetrievalResult(all_courses[1], 0.5, "fusion"),
        ]
        args = {
            "best_course_id": all_courses[1].id,
            "runner_up_id": all_courses[0].id,
            "reasoning": "Better fit for student goals.",
            "confidence": "high",
        }
        verdict = agent._build_verdict(args, results)
        assert verdict.best_course_id == all_courses[1].id
        assert verdict.runner_up_id == all_courses[0].id
        assert verdict.confidence == "high"

    def test_runner_up_hallucination_set_to_none(self, all_courses):
        agent = JudgeAgent()
        results = [RetrievalResult(all_courses[0], 0.9, "fusion")]
        args = {
            "best_course_id": all_courses[0].id,
            "runner_up_id": "FAKE_RUNNER",
            "reasoning": "reason",
            "confidence": "medium",
        }
        verdict = agent._build_verdict(args, results)
        assert verdict.runner_up_id is None


# ─────────────────────────────────────────────────────────────────────────────
#  10. ResponseAgent — output format
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseAgent:
    def _make_eligible(self, course_map, ids: list[str]) -> list[RetrievalResult]:
        return [RetrievalResult(course_map[cid], 0.5, "fusion") for cid in ids]

    def _make_locked(self, course_map, id: str, missing: list[str]) -> RetrievalResult:
        r = RetrievalResult(course_map[id], 0.3, "fusion")
        r.missing_prereqs = missing
        return r

    def test_output_contains_recommendation_header(
        self, response_agent, fresh_profile, course_map, all_courses
    ):
        eligible = self._make_eligible(course_map, ["CSIE4001"])
        verdict  = JudgeVerdict("CSIE4001", "Great fit.", "high")
        bm25     = self._make_eligible(course_map, ["CSIE4001"])
        vec      = self._make_eligible(course_map, ["CSIE4001"])
        out = response_agent.process(fresh_profile, eligible, [], bm25, vec, verdict, course_map)
        assert "TOP RECOMMENDATION" in out

    def test_output_shows_locked_courses(
        self, response_agent, beginner_profile, course_map
    ):
        eligible = self._make_eligible(course_map, ["CSIE1001"])
        locked   = [self._make_locked(course_map, "CSIE2001", ["CSIE1001"])]
        verdict  = JudgeVerdict("CSIE1001", "Best for beginners.", "high")
        bm25     = self._make_eligible(course_map, ["CSIE1001"])
        vec      = self._make_eligible(course_map, ["CSIE1001"])
        out = response_agent.process(beginner_profile, eligible, locked, bm25, vec, verdict, course_map)
        assert "LOCKED" in out
        assert "CSIE2001" in out

    def test_output_shows_no_eligible_message_when_empty(
        self, response_agent, beginner_profile, course_map
    ):
        locked = [self._make_locked(course_map, "CSIE2001", ["CSIE1001"])]
        out = response_agent.process(
            beginner_profile, [], locked, [], [], None, course_map
        )
        assert "NO ELIGIBLE" in out

    def test_student_profile_section_in_output(
        self, response_agent, fresh_profile, course_map
    ):
        eligible = self._make_eligible(course_map, ["CSIE4001"])
        verdict  = JudgeVerdict("CSIE4001", "Reason.", "medium")
        out = response_agent.process(fresh_profile, eligible, [], eligible, eligible, verdict, course_map)
        assert "STUDENT PROFILE" in out


# ─────────────────────────────────────────────────────────────────────────────
#  11. End-to-end pipeline (mocked Groq)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineEndToEnd:
    @patch("function.main.call_groq_with_tools")
    def test_beginner_gets_no_prereq_course(self, mock_groq):
        """Freshman with no completed courses should only get courses with no prereqs."""
        mock_groq.side_effect = [
            # IntakeAgent call
            {
                "academic_year": 1,
                "degree_level": "undergrad",
                "completed_courses": [],
                "goals": ["learn programming"],
                "constraints": [],
                "search_query": "introduction programming python beginner",
            },
            # JudgeAgent call
            {
                "best_course_id": "CSIE1001",
                "runner_up_id": "CSIE1002",
                "reasoning": "Best starting point for a beginner with no background.",
                "confidence": "high",
            },
        ]
        orchestrator = CourseFinderOrchestrator()
        output, profile = orchestrator.run("I am a first year student, want to take a programming course")

        assert profile is not None
        assert profile.academic_year == 1
        assert "CSIE1001" in output or "TOP RECOMMENDATION" in output

    @patch("function.main.call_groq_with_tools")
    def test_off_topic_returns_off_topic_message(self, mock_groq):
        orchestrator = CourseFinderOrchestrator()
        output, profile = orchestrator.run("Tell me a joke")
        assert output == OFF_TOPIC_RESPONSE
        assert profile is None
        mock_groq.assert_not_called()

    @patch("function.main.call_groq_with_tools")
    def test_senior_with_all_prereqs_gets_advanced_course(self, mock_groq):
        """Student who completed all prereqs should access deep learning."""
        completed = ["CSIE1001", "CSIE1002", "CSIE2001", "CSIE2002",
                     "CSIE3001", "MATH2001", "MATH2002", "CSIE4001"]
        mock_groq.side_effect = [
            {
                "academic_year": 4,
                "degree_level": "undergrad",
                "completed_courses": completed,
                "goals": ["deep learning", "computer vision"],
                "constraints": [],
                "search_query": "deep learning CNN transformer pytorch",
            },
            {
                "best_course_id": "CSIE4002",
                "runner_up_id": "CSIE4004",
                "reasoning": "Deep Learning is the natural next step after Machine Learning.",
                "confidence": "high",
            },
        ]
        orchestrator = CourseFinderOrchestrator()
        output, profile = orchestrator.run("I finished ML and want to go deeper into deep learning")

        assert profile is not None
        assert "CSIE4002" in output   # Deep Learning in output

    @patch("agents.JudgeAgent.call_groq_with_tools")
    @patch("agents.IntakeAgent.call_groq_with_tools")
    def test_profile_persists_across_turns(self, mock_intake, mock_judge):
        """Second turn should update existing profile, not create fresh one."""
        mock_intake.side_effect = [
            {
                "academic_year": 2,
                "completed_courses": ["CSIE1001"],
                "goals": ["data structures"],
                "constraints": [],
                "search_query": "data structures algorithms trees graphs",
            },
            {
                "academic_year": 2,
                "completed_courses": ["CSIE1001", "CSIE2001"],
                "goals": ["algorithms"],
                "constraints": [],
                "search_query": "algorithms divide conquer dynamic programming",
            },
        ]
        mock_judge.side_effect = [
            {
                "best_course_id": "CSIE2001",
                "reasoning": "Matches goals.",
                "confidence": "high",
            },
            {
                "best_course_id": "CSIE3001",
                "reasoning": "Ready for algorithms now.",
                "confidence": "high",
            },
        ]

        orchestrator = CourseFinderOrchestrator()
        _, profile = orchestrator.run("I want to learn data structures course")
        assert profile is not None
        assert "CSIE1001" in profile.completed_courses

        _, profile2 = orchestrator.run(
            "Now I want to study algorithms course", profile=profile
        )
        assert profile2 is not None
        assert "CSIE1001" in profile2.completed_courses
        assert "CSIE2001" in profile2.completed_courses

# ─────────────────────────────────────────────────────────────────────────────
#  12. UserProfile.describe()
# ─────────────────────────────────────────────────────────────────────────────

class TestUserProfileDescribe:
    def test_describe_contains_all_fields(self, fresh_profile):
        desc = fresh_profile.describe()
        assert "Degree" in desc
        assert "Year" in desc
        assert "Completed" in desc
        assert "Goals" in desc
        assert "Query" in desc

    def test_describe_empty_completed(self, beginner_profile):
        desc = beginner_profile.describe()
        assert "none" in desc.lower()

    def test_degree_label_undergrad(self, fresh_profile):
        assert "Undergraduate" in fresh_profile.describe()

    def test_degree_label_master(self):
        p = UserProfile("x", 5, "master", [], [], [], "test")
        assert "Master" in p.describe()

    def test_degree_label_phd(self):
        p = UserProfile("x", 7, "phd", [], [], [], "test")
        assert "PhD" in p.describe()


class TestAdminAPIAddCourse:
    def test_add_course_requires_admin_cookie(self):
        client = TestClient(app)
        response = client.post("/admin/add_course", json={"course": {"id": "TEST1001"}})
        assert response.status_code == 403

    def test_add_course_validates_required_fields(self):
        client = TestClient(app)
        response = client.post(
            "/admin/add_course",
            json={"course": {"id": "TEST1001", "name": "Demo Course"}},
            cookies={"admin": "true"},
        )
        assert response.status_code == 400
        assert "Missing required fields" in response.json()["detail"]

    def test_add_course_accepts_full_course_data(self):
        client = TestClient(app)
        new_course = {
            "id": "TEST1001",
            "name": "Demo Course",
            "credits": 3,
            "semester": "Fall",
            "schedule": "Monday 09:00-11:00",
            "instructor": "Prof. Demo",
            "prerequisites": [],
            "description": "A demo course for testing.",
            "department": "Test Department",
            "language": "English",
            "degree": "undergrad",
        }
        response = client.post(
            "/admin/add_course",
            json={"course": new_course},
            cookies={"admin": "true"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

        # cleanup added course so other tests remain deterministic
        RAW_COURSES[:] = [course for course in RAW_COURSES if course.get("id") != "TEST1001"]
        VALID_COURSE_IDS.discard("TEST1001")
        api_module.orchestrator = CourseFinderOrchestrator()


# ─────────────────────────────────────────────────────────────────────────────
#  13. Data integrity — RAW_COURSES
# ─────────────────────────────────────────────────────────────────────────────

class TestRawCourseData:
    def test_all_prereqs_reference_valid_ids(self):
        for c in RAW_COURSES:
            for p in c["prerequisites"]:
                assert p in VALID_COURSE_IDS, f"{c['id']} has unknown prereq {p}"

    def test_no_duplicate_course_ids(self):
        ids = [c["id"] for c in RAW_COURSES]
        assert len(ids) == len(set(ids))

    def test_all_required_fields_present(self):
        required = {"id", "name", "credits", "semester", "schedule",
                    "instructor", "prerequisites", "description", "department"}
        for c in RAW_COURSES:
            assert required.issubset(c.keys()), f"{c['id']} missing fields"

    def test_credits_are_positive_integers(self):
        for c in RAW_COURSES:
            assert isinstance(c["credits"], int) and c["credits"] > 0

    def test_course_full_text_not_empty(self, all_courses):
        for c in all_courses:
            assert len(c.full_text()) > 20