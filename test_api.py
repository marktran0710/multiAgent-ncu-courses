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


def make_profile(search_query: str = "machine learning") -> UserProfile:
    return UserProfile(
        raw_input="I want a course",
        academic_year=3,
        degree_level="undergrad",
        completed_courses=["CSIE1001"],
        goals=["machine learning"],
        constraints=[],
        search_query=search_query,
    )


class FakeOrchestrator:
    def __init__(self, profile: UserProfile | None = None):
        self.profile = profile or make_profile()
        self.calls = []
        self.course_map = {
            "CSIE4001": Course(**{
                "id": "CSIE4001",
                "name": "Machine Learning",
                "credits": 3,
                "semester": "Fall / Spring",
                "schedule": "Tuesday",
                "instructor": "Prof. Demo",
                "prerequisites": [],
                "description": "Machine learning course",
                "department": "CSIE",
                "language": "Chinese",
            })
        }

    def run_user(self, message: str, profile: UserProfile | None = None):
        self.calls.append((message, profile))
        details = {
            "full_output": "full response",
            "eligible": [{"id": "CSIE4001"}],
            "locked": [],
            "query_rag": [{"id": "CSIE4001"}],
            "retrieval_methods": {"bm25_agent": [{"id": "CSIE4001"}]},
            "verdict": {"best_course_id": "CSIE4001"},
            "response_evaluation": {"status": "completed", "approved": True},
            "score_explanation": api_module.SCORE_EXPLANATION,
        }
        return "I recommend CSIE4001.", self.profile, details


@pytest.fixture
def isolated_api_state(monkeypatch):
    fake = FakeOrchestrator()
    monkeypatch.setattr(api_module, "orchestrator", fake)
    api_module.user_sessions.clear()
    api_module.chat_histories.clear()
    api_module.last_recommendations.clear()
    api_module.conversation_logs.clear()
    api_module.admin_sessions.clear()
    api_module.clear_benchmark_cache()
    yield fake
    api_module.user_sessions.clear()
    api_module.chat_histories.clear()
    api_module.last_recommendations.clear()
    api_module.conversation_logs.clear()
    api_module.admin_sessions.clear()
    api_module.clear_benchmark_cache()


class TestChatLogic:
    def test_serialize_profile_handles_none_and_profile(self):
        profile = make_profile("nlp")
        assert api_module.serialize_profile(None) is None
        payload = api_module.serialize_profile(profile)
        assert payload["search_query"] == "nlp"
        assert payload["degree_level"] == "undergrad"

    def test_run_chat_message_updates_session_and_log(self, isolated_api_state):
        output, profile, details = api_module.run_chat_message("s1", "I want ML")

        assert output == "I recommend CSIE4001."
        assert api_module.user_sessions["s1"] is profile
        assert api_module.last_recommendations["s1"] == "CSIE4001"
        assert details["verdict"]["best_course_id"] == "CSIE4001"
        assert api_module.conversation_logs[-1]["session_id"] == "s1"
        assert api_module.chat_histories["s1"] == [
            {"sender": "user", "text": "I want ML"},
            {"sender": "bot", "text": "I recommend CSIE4001."},
        ]
        assert api_module.conversation_logs[-1]["retrieval_methods"]["bm25_agent"][0]["id"] == "CSIE4001"
        assert api_module.conversation_logs[-1]["response_evaluation"]["approved"] is True
        assert "Raw retrieval scores use different units" in api_module.conversation_logs[-1]["score_explanation"]

    def test_run_chat_message_passes_existing_profile(self, isolated_api_state):
        existing = make_profile("existing")
        api_module.user_sessions["s1"] = existing

        api_module.run_chat_message("s1", "follow up")

        assert isolated_api_state.calls[-1] == ("follow up", existing)

    def test_language_follow_up_answers_previous_recommended_course(self, isolated_api_state):
        api_module.run_chat_message("s1", "I want ML")

        output, profile, details = api_module.run_chat_message(
            "s1",
            "ok but the course taught by english or chinese",
        )

        assert output == "[CSIE4001] Machine Learning is taught in Chinese."
        assert details["verdict"]["best_course_id"] == "CSIE4001"
        assert len(isolated_api_state.calls) == 1
        assert api_module.conversation_logs[-1]["bot_response"] == output
        assert api_module.chat_histories["s1"][-2:] == [
            {"sender": "user", "text": "ok but the course taught by english or chinese"},
            {"sender": "bot", "text": "[CSIE4001] Machine Learning is taught in Chinese."},
        ]
        assert api_module.conversation_logs[-1]["score_explanation"] == api_module.SCORE_EXPLANATION

    def test_language_follow_up_without_previous_course_uses_normal_chat(self, isolated_api_state):
        api_module.run_chat_message("s1", "is the course taught in english or chinese")
        assert len(isolated_api_state.calls) == 1

    def test_english_taught_course_request_is_not_previous_course_clarification(self, isolated_api_state):
        api_module.run_chat_message("s1", "I want ML")
        api_module.run_chat_message("s1", "any course is taught by english I can learn")

        assert len(isolated_api_state.calls) == 2
        assert isolated_api_state.calls[-1][0] == "any course is taught by english I can learn"

    def test_language_answer_returns_none_when_course_metadata_missing(self, isolated_api_state):
        api_module.last_recommendations["s1"] = "UNKNOWN"
        assert api_module.answer_course_language("s1") is None

    def test_http_chat_sets_cookie_and_returns_profile(self, isolated_api_state):
        client = TestClient(app)
        response = client.post("/chat", json={"message": "I want ML"})

        assert response.status_code == 200
        assert "session_id" in response.cookies
        assert response.json()["response"] == "I recommend CSIE4001."
        assert response.json()["profile"]["search_query"] == "machine learning"
        assert response.json()["history"][-1]["text"] == "I recommend CSIE4001."

    def test_chat_history_endpoint_uses_session_query_param(self, isolated_api_state):
        client = TestClient(app)
        api_module.run_chat_message("known-session", "I want ML")

        response = client.get("/chat/history?session_id=known-session")

        assert response.status_code == 200
        assert response.json()["session_id"] == "known-session"
        assert response.json()["profile"]["search_query"] == "machine learning"
        assert response.json()["history"][0] == {"sender": "user", "text": "I want ML"}

    def test_profile_endpoint_returns_404_without_session(self, isolated_api_state):
        client = TestClient(app)
        response = client.get("/profile")
        assert response.status_code == 404

    def test_profile_endpoint_returns_existing_profile(self, isolated_api_state):
        client = TestClient(app)
        api_module.user_sessions["known"] = make_profile("vision")
        response = client.get("/profile?session_id=known")
        assert response.status_code == 200
        assert response.json()["search_query"] == "vision"

    def test_realtime_websocket_session_empty_message_and_response(self, isolated_api_state):
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=ws-test") as websocket:
            session_event = websocket.receive_json()
            assert session_event["type"] == "session"
            assert session_event["session_id"] == "ws-test"
            assert session_event["history"] == []

            websocket.send_json({"message": "   "})
            error_event = websocket.receive_json()
            assert error_event["type"] == "error"

            websocket.send_json({"message": "I want ML"})
            status_event = websocket.receive_json()
            response_event = websocket.receive_json()
            done_event = websocket.receive_json()

            assert status_event["type"] == "status"
            assert response_event["type"] == "response"
            assert response_event["response"] == "I recommend CSIE4001."
            assert response_event["verdict"]["best_course_id"] == "CSIE4001"
            assert response_event["response_evaluation"]["approved"] is True
            assert done_event["type"] == "done"

        with client.websocket_connect("/ws/chat?session_id=ws-test") as websocket:
            restored_session = websocket.receive_json()
            assert restored_session["type"] == "session"
            assert restored_session["history"][-1]["text"] == "I recommend CSIE4001."


def valid_course_payload(course_id: str = "TEST9001") -> dict:
    return {
        "id": course_id,
        "name": "International AI Seminar",
        "credits": 3,
        "semester": "Fall",
        "schedule": "Monday 09:00-11:00",
        "instructor": "Prof. Demo",
        "prerequisites": [],
        "description": "A course for international advising tests.",
        "department": "Computer Science and Information Engineering",
        "language": "english",
        "degree": "UNDERGRAD",
    }


def admin_cookie() -> dict:
    token = "test-admin-session"
    api_module.admin_sessions.add(token)
    return {"admin_session": token}


class FakeCourseFinderOrchestrator:
    def __init__(self):
        course = Course(**{
            "id": "CSIE1001",
            "name": "Introduction to Programming",
            "credits": 3,
            "semester": "Fall",
            "schedule": "Monday",
            "instructor": "Prof. Demo",
            "prerequisites": [],
            "description": "Intro course",
            "department": "CSIE",
        })
        self.bm25_agent = type("FakeBM25", (), {"courses": [course]})()


class FakeRetrievalBenchmarkAgent:
    run_count = 0

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def run(self):
        type(self).run_count += 1
        return {
            "case_count": 1,
            "top_k": 10,
            "comparison": {
                "legacy_best": {"label": "Past BM25 + Vector FusionAgent", "recall_at_3": 1.0},
                "current": {"label": "Final weighted RRF fusion", "recall_at_3": 1.0},
                "delta": {"recall_at_3": 0.0},
                "percentage_point_change": {"recall_at_3": 0.0},
                "relative_percent_change": {"recall_at_3": 0.0},
                "case_counts": {"improved": 0, "same": 1, "regressed": 0},
                "special_cases": [],
            },
            "summary": [{"method": "bm25_agent", "label": "BM25Agent", "recall_at_3": 1.0}],
            "cases": [],
        }


class TestApiRoutesAndAdminLogic:
    def test_static_pages_render(self):
        client = TestClient(app)
        assert client.get("/").status_code == 200
        assert client.get("/admin").status_code == 200

    def test_healthz_endpoint(self):
        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_admin_login_success_and_failure(self):
        client = TestClient(app)
        success = client.post("/admin/login", json={"password": "admin123"})
        failure = client.post("/admin/login", json={"password": "wrong"})

        assert success.status_code == 200
        assert success.json()["success"] is True
        assert "admin_session" in success.cookies
        assert failure.status_code == 401

    def test_admin_status_and_logout(self):
        client = TestClient(app)
        anonymous = client.get("/admin/status")
        login = client.post("/admin/login", json={"password": "admin123"})
        logged_in = client.get("/admin/status")
        logout = client.post("/admin/logout")
        logged_out = client.get("/admin/status")

        assert anonymous.status_code == 200
        assert anonymous.json() == {"authenticated": False, "bypass_available": False}
        assert login.status_code == 200
        assert logged_in.status_code == 200
        assert logged_in.json() == {"authenticated": True, "bypass_available": False}
        assert logout.status_code == 200
        assert logout.json()["success"] is True
        assert logged_out.json() == {"authenticated": False, "bypass_available": False}

    def test_admin_dev_login_requires_bypass_flag(self, monkeypatch):
        client = TestClient(app)
        disabled = client.post("/admin/dev-login")

        monkeypatch.setattr(api_module, "ADMIN_BYPASS_ENABLED", True)
        enabled = client.post("/admin/dev-login")
        status = client.get("/admin/status")

        assert disabled.status_code == 403
        assert enabled.status_code == 200
        assert enabled.json()["success"] is True
        assert status.json() == {"authenticated": True, "bypass_available": True}

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("id", "", "Course ID"),
            ("name", "", "Course name"),
            ("semester", "", "Semester"),
            ("schedule", "", "Schedule"),
            ("instructor", "", "Instructor"),
            ("description", "", "Description"),
            ("department", "", "Department"),
            ("credits", 0, "Credits"),
            ("prerequisites", "CSIE1001", "Prerequisites"),
            ("language", "French", "Language"),
            ("degree", "certificate", "Degree"),
        ],
    )
    def test_add_course_validation_branches(self, field, value, message):
        client = TestClient(app)
        payload = valid_course_payload(f"TEST_{field.upper()}")
        payload[field] = value

        response = client.post(
            "/admin/add_course",
            json={"course": payload},
            cookies=admin_cookie(),
        )

        assert response.status_code == 400
        assert message.lower() in response.json()["detail"].lower()

    def test_add_course_rejects_duplicate_id(self):
        client = TestClient(app)
        payload = valid_course_payload("CSIE1001")
        response = client.post(
            "/admin/add_course",
            json={"course": payload},
            cookies=admin_cookie(),
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_add_course_rejects_unknown_prerequisite(self):
        client = TestClient(app)
        payload = valid_course_payload("TEST_BAD_PREREQ")
        payload["prerequisites"] = ["NOPE1001"]
        response = client.post(
            "/admin/add_course",
            json={"course": payload},
            cookies=admin_cookie(),
        )
        assert response.status_code == 400
        assert "Unknown prerequisite" in response.json()["detail"]

    def test_add_course_success_normalizes_language_and_degree(self, monkeypatch):
        client = TestClient(app)
        monkeypatch.setattr(api_module, "CourseFinderOrchestrator", FakeCourseFinderOrchestrator)
        payload = valid_course_payload("TEST_SUCCESS")

        response = client.post(
            "/admin/add_course",
            json={"course": payload},
            cookies=admin_cookie(),
        )

        try:
            assert response.status_code == 200
            assert response.json()["success"] is True
            added = next(course for course in RAW_COURSES if course["id"] == "TEST_SUCCESS")
            assert added["language"] == "English"
            assert added["degree"] == "undergrad"
        finally:
            RAW_COURSES[:] = [course for course in RAW_COURSES if course.get("id") != "TEST_SUCCESS"]
            VALID_COURSE_IDS.discard("TEST_SUCCESS")

    def test_update_data_requires_admin_and_succeeds_with_admin(self, monkeypatch):
        client = TestClient(app)
        monkeypatch.setattr(api_module, "CourseFinderOrchestrator", FakeCourseFinderOrchestrator)

        denied = client.post("/admin/update_data")
        allowed = client.post("/admin/update_data", cookies=admin_cookie())

        assert denied.status_code == 403
        assert allowed.status_code == 200
        assert allowed.json()["success"] is True

    def test_admin_logs_require_admin_and_return_logs(self):
        client = TestClient(app)
        api_module.conversation_logs[:] = [{"session_id": "s1"}]

        denied = client.get("/admin/logs")
        allowed = client.get("/admin/logs", cookies=admin_cookie())

        assert denied.status_code == 403
        assert allowed.status_code == 200
        assert allowed.json() == [{"session_id": "s1"}]

    def test_admin_benchmark_requires_admin_and_returns_metrics(self, monkeypatch):
        client = TestClient(app)
        monkeypatch.setattr(api_module, "RetrievalBenchmarkAgent", FakeRetrievalBenchmarkAgent)
        api_module.clear_benchmark_cache()
        FakeRetrievalBenchmarkAgent.run_count = 0

        denied = client.get("/admin/benchmark")
        allowed = client.get("/admin/benchmark", cookies=admin_cookie())

        assert denied.status_code == 403
        assert allowed.status_code == 200
        assert allowed.json()["summary"][0]["recall_at_3"] == 1.0
        assert allowed.json()["comparison"]["legacy_best"]["label"] == "Past BM25 + Vector FusionAgent"
        assert allowed.json()["comparison"]["current"]["label"] == "Final weighted RRF fusion"
        assert FakeRetrievalBenchmarkAgent.run_count == 1

    def test_public_benchmark_page_and_data_render(self, monkeypatch):
        client = TestClient(app)
        monkeypatch.setattr(api_module, "RetrievalBenchmarkAgent", FakeRetrievalBenchmarkAgent)
        api_module.clear_benchmark_cache()
        FakeRetrievalBenchmarkAgent.run_count = 0

        page = client.get("/benchmark")
        data = client.get("/benchmark/data")
        cached_data = client.get("/benchmark/data")
        cached_page = client.get("/benchmark")

        assert page.status_code == 200
        assert "Old FusionAgent vs Current Query RAG" in page.text
        assert "window.__BENCHMARK_DATA__ = null;" in page.text
        assert data.status_code == 200
        assert data.json()["comparison"]["percentage_point_change"]["recall_at_3"] == 0.0
        assert cached_data.status_code == 200
        assert cached_data.json()["case_count"] == 1
        assert cached_page.status_code == 200
        assert "window.__BENCHMARK_DATA__ = null;" not in cached_page.text
        assert '"case_count": 1' in cached_page.text
        assert FakeRetrievalBenchmarkAgent.run_count == 1

    def test_admin_courses_require_admin_and_return_courses(self, monkeypatch):
        client = TestClient(app)
        monkeypatch.setattr(api_module, "orchestrator", FakeCourseFinderOrchestrator())

        denied = client.get("/admin/courses")
        allowed = client.get("/admin/courses", cookies=admin_cookie())

        assert denied.status_code == 403
        assert allowed.status_code == 200
        assert allowed.json()[0]["id"] == "CSIE1001"


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
            cookies=admin_cookie(),
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
            cookies=admin_cookie(),
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

        # cleanup added course so other tests remain deterministic
        RAW_COURSES[:] = [course for course in RAW_COURSES if course.get("id") != "TEST1001"]
        VALID_COURSE_IDS.discard("TEST1001")
        api_module.orchestrator = api_module.CourseFinderOrchestrator()
