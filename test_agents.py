"""
Tests for agents: BM25Agent, VectorAgent, FusionAgent, IntakeAgent, JudgeAgent, ResponseAgent
Run: pytest test_agents.py -v
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
#  Import agents and related
# ─────────────────────────────────────────────────────────────────────────────
from models.Course import Course
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile
from models.RetrievalResult import RetrievalResult
from models.JudgeVerdict import JudgeVerdict
from keywords.CourseKeywords import COURSE_KEYWORDS
from function.main import tokenize, reciprocal_rank_fusion, check_prerequisites_met
from agents.IntakeAgent import IntakeAgent
from agents.BM25 import BM25Agent
from agents.QueryRAGAgent import QueryRAGAgent
from agents.VectorAgent import VectorAgent
from agents.FusionAgent import FusionAgent
from agents.JudgeAgent import JudgeAgent
from agents.ResponseEvaluationAgent import ResponseEvaluationAgent
from agents.ResponseAgent import ResponseAgent


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
def query_rag_agent(all_courses) -> QueryRAGAgent:
    return QueryRAGAgent(all_courses)

@pytest.fixture
def fusion_agent() -> FusionAgent:
    return FusionAgent()

@pytest.fixture
def response_agent() -> ResponseAgent:
    return ResponseAgent()


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

    def test_transformer_load_failure_uses_tfidf_fallback(self, monkeypatch, all_courses, fresh_profile):
        import agents.VectorAgent as vector_module

        def fail_to_load(*args, **kwargs):
            raise RuntimeError("offline model cache")

        monkeypatch.setattr(vector_module, "_ST_AVAILABLE", True)
        monkeypatch.setattr(vector_module, "SentenceTransformer", fail_to_load)

        agent = VectorAgent(all_courses)
        results = agent.process(fresh_profile, top_k=3)

        assert agent._use_transformer is False
        assert agent.collection is None
        assert len(results) == 3


# ─────────────────────────────────────────────────────────────────────────────
#  6. FusionAgent — prerequisite filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryRAGAgent:
    def test_runs_all_requested_strategies(self, query_rag_agent, fresh_profile):
        rankings = query_rag_agent.process(fresh_profile, top_k=3)
        assert set(rankings) == {strategy.key for strategy in QueryRAGAgent.STRATEGIES}
        assert all(len(results) == 3 for results in rankings.values())

    def test_strongest_sparse_best_fields_is_available(self, query_rag_agent, fresh_profile):
        rankings = query_rag_agent.process(fresh_profile, top_k=5)
        ids = [r.course.id for r in rankings["sparse_best_fields"]]
        assert "CSIE4001" in ids

    def test_query_rag_fusion_returns_ranked_results(self, query_rag_agent, fresh_profile):
        rankings = query_rag_agent.process(fresh_profile, top_k=4)
        fused = query_rag_agent.fuse(rankings)
        assert fused
        assert all(r.source == "query_rag" for r in fused)
        assert [r.score for r in fused] == sorted([r.score for r in fused], reverse=True)

    def test_query_rag_fusion_accepts_original_bm25_and_vector_paths(self, query_rag_agent, fresh_profile):
        rankings = query_rag_agent.process(fresh_profile, top_k=4)
        rankings["bm25_agent"] = rankings["bm25_multi_match"]
        rankings["vector_agent"] = rankings["knn_multi_match"]
        fused = query_rag_agent.fuse(rankings)
        assert fused


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

    def test_language_preference_prioritizes_without_filtering(self, fusion_agent):
        profile = UserProfile(
            raw_input="I prefer English, but Chinese is okay",
            academic_year=1,
            degree_level="undergrad",
            completed_courses=[],
            goals=["programming"],
            constraints=["English courses preferred"],
            preferred_language="English",
            language_priority="preferred",
            search_query="programming",
        )
        chinese = Course(**RAW_COURSES[0], language="Chinese")
        english = Course(**RAW_COURSES[1], language="English")
        results = [
            RetrievalResult(chinese, 0.9, "fusion"),
            RetrievalResult(english, 0.1, "fusion"),
        ]

        eligible, locked = fusion_agent.filter_results(results, profile)

        assert locked == []
        assert [r.course.language for r in eligible] == ["English", "Chinese"]

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

    def test_heuristic_fallback_keeps_language_hint(self):
        agent = IntakeAgent()
        profile = agent._heuristic_fallback("any course is taught by english I can learn")
        assert profile.preferred_language == "English"
        assert profile.language_priority == "required"
        assert "English courses only" in profile.constraints

    def test_process_llm_failure_updates_existing_profile_with_language_hint(self, monkeypatch):
        agent = IntakeAgent()
        existing = UserProfile(
            raw_input="old",
            academic_year=3,
            degree_level="undergrad",
            completed_courses=["CSIE1001", "CSIE2001"],
            goals=["machine learning"],
            constraints=[],
            search_query="machine learning",
        )

        def raise_error(_messages):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(agent, "_call_llm", raise_error)
        profile = agent.process("any course is taught by english I can learn", existing_profile=existing)

        assert profile is existing
        assert profile.preferred_language == "English"
        assert profile.language_priority == "required"
        assert "English courses only" in profile.constraints

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

    def test_apply_language_hint_adds_english_constraint(self):
        args = {"constraints": [], "search_query": "english taught courses"}
        updated = IntakeAgent._apply_language_hint(
            "any course is taught by english I can learn",
            args,
        )
        assert "English courses only" in updated["constraints"]

    def test_apply_language_hint_adds_soft_english_preference(self):
        args = {"constraints": [], "search_query": "english courses"}
        updated = IntakeAgent._apply_language_hint(
            "I prefer English courses",
            args,
        )
        assert "English courses preferred" in updated["constraints"]

    def test_apply_language_hint_does_not_convert_language_comparison_question(self):
        args = {"constraints": [], "search_query": "course language"}
        updated = IntakeAgent._apply_language_hint(
            "ok but the course taught by english or chinese",
            args,
        )
        assert updated["constraints"] == []

    def test_extract_language_priority_levels(self):
        assert UserProfile._extract_language_preference(["English courses only"]) == (
            "English",
            "required",
        )
        assert UserProfile._extract_language_preference(["Chinese courses preferred"]) == (
            "Chinese",
            "preferred",
        )

    def test_call_llm_falls_back_to_gemini_when_groq_fails(self, monkeypatch):
        agent = IntakeAgent(provider="groq")

        def fail_groq(*_args, **_kwargs):
            raise RuntimeError("groq unavailable")

        gemini = MagicMock(return_value={
            "academic_year": 1,
            "completed_courses": [],
            "goals": ["programming"],
            "constraints": [],
            "search_query": "programming",
        })

        monkeypatch.setattr("agents.IntakeAgent.call_groq_with_tools", fail_groq)
        monkeypatch.setattr("agents.IntakeAgent.call_gemini_with_tools", gemini)

        result = agent._call_llm([{"role": "user", "content": "course"}])
        assert result["search_query"] == "programming"
        gemini.assert_called_once()


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
        assert "Groq" not in verdict.reasoning
        assert "unavailable" not in verdict.reasoning

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

    def test_call_llm_falls_back_to_gemini_when_groq_fails(self, monkeypatch, all_courses):
        agent = JudgeAgent(provider="groq")

        def fail_groq(*_args, **_kwargs):
            raise RuntimeError("groq unavailable")

        gemini = MagicMock(return_value={
            "best_course_id": all_courses[0].id,
            "runner_up_id": all_courses[1].id,
            "reasoning": "Gemini fallback selected the best eligible course.",
            "confidence": "medium",
        })

        monkeypatch.setattr("agents.JudgeAgent.call_groq_with_tools", fail_groq)
        monkeypatch.setattr("agents.JudgeAgent.call_gemini_with_tools", gemini)

        result = agent._call_llm([{"role": "user", "content": "judge"}])
        assert result["best_course_id"] == all_courses[0].id
        gemini.assert_called_once()


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

    def test_minimal_response_uses_locked_english_pathway(
        self, response_agent, course_map
    ):
        profile = UserProfile(
            raw_input="Recommend an English-taught AI course",
            academic_year=1,
            degree_level="undergrad",
            completed_courses=[],
            goals=["AI"],
            constraints=["English courses only"],
            preferred_language="English",
            search_query="English taught AI course",
        )
        chinese_locked = self._make_locked(course_map, "CSIE4001", ["CSIE3001", "MATH2001"])
        english_locked = self._make_locked(course_map, "CSIE4005", ["CSIE4001"])

        out = response_agent.minimal_response(
            None,
            course_map,
            locked_results=[chinese_locked, english_locked],
            profile=profile,
        )

        assert "[CSIE4005] Reinforcement Learning" in out
        assert "Complete first: CSIE4001 (Machine Learning)." in out
        assert "I couldn't find an eligible course" not in out

    def test_student_profile_section_in_output(
        self, response_agent, fresh_profile, course_map
    ):
        eligible = self._make_eligible(course_map, ["CSIE4001"])
        verdict  = JudgeVerdict("CSIE4001", "Reason.", "medium")
        out = response_agent.process(fresh_profile, eligible, [], eligible, eligible, verdict, course_map)
        assert "STUDENT PROFILE" in out


class TestResponseEvaluationAgent:
    def test_uses_different_provider_from_primary(self):
        assert ResponseEvaluationAgent(primary_provider="groq").provider == "gemini"
        assert ResponseEvaluationAgent(primary_provider="gemini").provider == "groq"

    def test_skips_when_provider_key_is_not_configured(self, monkeypatch, fresh_profile):
        monkeypatch.setenv("GEMINI_API_KEY", "placeholder")
        monkeypatch.setenv("GROQ_API_KEY", "placeholder")
        agent = ResponseEvaluationAgent(primary_provider="groq")

        response, metadata = agent.process(
            draft_response="Draft answer",
            profile=fresh_profile,
            eligible=[],
            locked=[],
            verdict={},
        )

        assert response == "Draft answer"
        assert metadata["status"] == "skipped"
        assert metadata["approved"] is True

    def test_uses_revised_response_when_not_approved(self, monkeypatch, fresh_profile):
        monkeypatch.setenv("GEMINI_API_KEY", "configured-test-key")
        agent = ResponseEvaluationAgent(primary_provider="groq")
        evaluator = MagicMock(return_value={
            "approved": False,
            "score": 2,
            "issues": ["Locked course was phrased as eligible."],
            "revised_response": "Closest match is locked; complete prerequisites first.",
        })
        monkeypatch.setattr(agent, "_call_llm", evaluator)

        response, metadata = agent.process(
            draft_response="Take this locked course now.",
            profile=fresh_profile,
            eligible=[],
            locked=[{"id": "CSIE4001", "missing_prereqs": ["CSIE3001"]}],
            verdict={"best_course_id": None},
        )

        assert response == "Closest match is locked; complete prerequisites first."
        assert metadata["status"] == "completed"
        assert metadata["approved"] is False
        assert metadata["score"] == 2
        assert metadata["provider"] == "gemini"

    def test_falls_back_to_alternate_groq_model_when_gemini_fails(self, monkeypatch, fresh_profile):
        monkeypatch.setenv("GEMINI_API_KEY", "configured-test-key")
        monkeypatch.setenv("GROQ_API_KEY", "configured-test-key")
        agent = ResponseEvaluationAgent(primary_provider="groq", primary_model="llama-3.3-70b-versatile")

        def evaluator(_messages, provider, model):
            if provider == "gemini":
                raise RuntimeError("quota exhausted")
            return {
                "approved": True,
                "score": 5,
                "issues": [],
                "revised_response": "Draft answer",
            }

        monkeypatch.setattr(agent, "_call_llm", evaluator)

        response, metadata = agent.process(
            draft_response="Draft answer",
            profile=fresh_profile,
            eligible=[],
            locked=[],
            verdict={},
        )

        assert response == "Draft answer"
        assert metadata["status"] == "completed"
        assert metadata["provider"] == "groq"
        assert metadata["model"] == "llama-3.1-8b-instant"

    def test_parses_groq_failed_generation_payload(self):
        error_text = (
            "tool_use_failed failed_generation: "
            '<function=evaluate_course_response>{"approved": false, "score": 4, '
            '"issues": ["needs prerequisite wording"], '
            '"revised_response": "Complete prerequisites first.", '
            '"extra_field": []}</function>'
        )

        parsed = ResponseEvaluationAgent._parse_groq_failed_generation(error_text)

        assert parsed == {
            "approved": False,
            "score": 4,
            "issues": ["needs prerequisite wording"],
            "revised_response": "Complete prerequisites first.",
        }

    def test_parses_groq_failed_generation_with_escaped_single_quotes(self):
        error_text = (
            '<function=evaluate_course_response>{"approved": false, "score": 2, '
            '"issues": ["locked course"], '
            '"locked_courses": "[{\\\\\\\'id\\\\\\\': \\\\\\\'CSIE4010\\\\\\\'}]", '
            '"revised_response": "Complete CSIE3008 first."}</function>'
        )

        parsed = ResponseEvaluationAgent._parse_groq_failed_generation(error_text)

        assert parsed["approved"] is False
        assert parsed["revised_response"] == "Complete CSIE3008 first."
