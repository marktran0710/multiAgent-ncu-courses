"""
Tests for utilities: tokenize, check_prerequisites_met, reciprocal_rank_fusion
Run: pytest test_utilities.py -v
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
#  Import utilities
# ─────────────────────────────────────────────────────────────────────────────
from models.Course import Course
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile
from models.RetrievalResult import RetrievalResult
from function.main import tokenize, reciprocal_rank_fusion, check_prerequisites_met


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def all_courses() -> list[Course]:
    return [Course(**c) for c in RAW_COURSES]

@pytest.fixture
def course_map(all_courses) -> dict[str, Course]:
    return {c.id: c for c in all_courses}


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