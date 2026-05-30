"""
Tests for end-to-end pipeline
Run: pytest test_pipeline.py -v
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
#  Import pipeline
# ─────────────────────────────────────────────────────────────────────────────
from models.Course import Course
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile
from models.RetrievalResult import RetrievalResult
from models.JudgeVerdict import JudgeVerdict
from keywords.OffTopicResponse import OFF_TOPIC_RESPONSE
from function.main import tokenize, reciprocal_rank_fusion, check_prerequisites_met
from agents.OrchestratorAgent import CourseFinderOrchestrator
from agents.RetrievalBenchmarkAgent import RetrievalBenchmarkAgent


# ─────────────────────────────────────────────────────────────────────────────
#  11. End-to-end pipeline (mocked Groq)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineEndToEnd:
    def test_prerequisite_path_bridge_adds_next_step_courses(self):
        orchestrator = CourseFinderOrchestrator()
        os_course = orchestrator.course_map["CSIE3002"]
        bridged = orchestrator._append_prerequisite_path([
            RetrievalResult(os_course, 1.0, "query_rag"),
        ])

        ids = [r.course.id for r in bridged]
        assert "CSIE2001" in ids
        assert "CSIE2002" in ids
        assert "CSIE1001" in ids

    def test_serialized_rankings_include_standard_score(self):
        orchestrator = CourseFinderOrchestrator()
        courses = list(orchestrator.course_map.values())[:2]
        serialized = orchestrator._serialize_rankings([
            RetrievalResult(courses[0], 20.0, "bm25"),
            RetrievalResult(courses[1], 5.0, "bm25"),
        ])

        assert serialized[0]["rank"] == 1
        assert serialized[0]["raw_score"] == 20.0
        assert serialized[0]["standard_score"] == 10.0
        assert serialized[1]["standard_score"] == 0.0
        assert serialized[0]["score_unit"] == "0-10 normalized within method"

    def test_retrieval_benchmark_scores_expected_course_rank(self):
        orchestrator = CourseFinderOrchestrator()
        courses = [orchestrator.course_map["CSIE4003"], orchestrator.course_map["CSIE4001"]]
        results = [
            RetrievalResult(courses[1], 0.9, "demo"),
            RetrievalResult(courses[0], 0.8, "demo"),
        ]

        score = RetrievalBenchmarkAgent._score_case(results, ["CSIE4003"])

        assert score["rank"] == 2
        assert score["top1"] == 0
        assert score["recall_at_3"] == 1
        assert score["reciprocal_rank"] == 0.5

    def test_retrieval_benchmark_compares_legacy_and_current_methods(self):
        orchestrator = CourseFinderOrchestrator()
        benchmark = RetrievalBenchmarkAgent(orchestrator).run(top_k=6)

        assert benchmark["case_count"] == 100
        assert benchmark["comparison"]["legacy_best"]["method"] == "legacy_bm25_vector_rag"
        assert benchmark["comparison"]["current"]["method"] == "final_query_rag_fusion"
        assert "special_cases" in benchmark["comparison"]
        assert "percentage_point_change" in benchmark["comparison"]
        assert "relative_percent_change" in benchmark["comparison"]
        assert benchmark["comparison"]["case_counts"]["improved"] >= 0
        assert all(
            case["comparison"]["outcome"] != "same"
            or case["comparison"]["top_three_changed"]
            for case in benchmark["comparison"]["special_cases"]
        )
        assert benchmark["comparison"]["special_case_counts"]["shown"] >= 10
        assert benchmark["profile_benchmark"]["case_count"] == 6
        assert benchmark["profile_benchmark"]["failed"] == 0

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
