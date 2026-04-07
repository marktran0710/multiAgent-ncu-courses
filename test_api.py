"""
Tests for API endpoints
Run: pytest test_api.py -v
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
#  Import API
# ─────────────────────────────────────────────────────────────────────────────
from models.Course import Course
from models.UserProfile import DEGREE_YEAR_RANGES, RAW_COURSES, VALID_COURSE_IDS, UserProfile
from fastapi.testclient import TestClient
import api as api_module
from api import app


# ─────────────────────────────────────────────────────────────────────────────
#  14. Admin API Add Course
# ─────────────────────────────────────────────────────────────────────────────

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
        api_module.orchestrator = api_module.CourseFinderOrchestrator()