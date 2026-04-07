"""
Tests for models: UserProfile, Course data integrity
Run: pytest test_models.py -v
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
#  Import models
# ─────────────────────────────────────────────────────────────────────────────
from models.Course import Course
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def all_courses() -> list[Course]:
    return [Course(**c) for c in RAW_COURSES]

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