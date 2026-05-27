from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import json
import os
import secrets
from typing import Optional, Dict, Any
from agents.OrchestratorAgent import CourseFinderOrchestrator
from agents.RetrievalBenchmarkAgent import RetrievalBenchmarkAgent
from models.UserProfile import RAW_COURSES, UserProfile, VALID_COURSE_IDS
from models.Course import Course

app = FastAPI()

# Global orchestrator
orchestrator = CourseFinderOrchestrator()

# In-memory storage for simplicity (use database in production)
user_sessions: Dict[str, UserProfile] = {}
last_recommendations: Dict[str, str] = {}
conversation_logs: list = []
admin_sessions: set[str] = set()

# Admin password (override this before publishing beyond a demo tunnel)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
ADMIN_BYPASS_ENABLED = os.environ.get("ADMIN_BYPASS_ENABLED", "").lower() in {"1", "true", "yes"}
SCORE_EXPLANATION = (
    "Raw retrieval scores use different units by algorithm. "
    "standard_score is normalized to 0-10 within each method for display; "
    "final fusion uses weighted Reciprocal Rank Fusion over ranks."
)

class ChatRequest(BaseModel):
    message: str

class AddCourseRequest(BaseModel):
    course: Dict[str, Any]

class LoginRequest(BaseModel):
    password: str

def serialize_profile(profile: Optional[UserProfile]) -> Optional[dict]:
    return profile.__dict__ if profile else None

def is_admin_request(req: Request) -> bool:
    token = req.cookies.get("admin_session")
    return bool(token and token in admin_sessions)

def create_admin_session(response: Response) -> None:
    token = secrets.token_urlsafe(32)
    admin_sessions.add(token)
    response.set_cookie(
        key="admin_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 8,
    )
    response.delete_cookie(key="admin")

def is_course_language_question(message: str) -> bool:
    text = message.lower()
    has_language = any(word in text for word in ("english", "chinese", "language", "taught"))
    asks_previous_course = any(
        phrase in text
        for phrase in (
            "the course",
            "this course",
            "that course",
            "recommended course",
            "the recommendation",
            "this recommendation",
            "that recommendation",
        )
    ) or any(f" {pronoun} " in f" {text} " for pronoun in ("it", "this", "that"))
    asks_for_any_course = any(
        phrase in text
        for phrase in (
            "any course",
            "which course",
            "what course",
            "courses",
            "can learn",
            "can i take",
            "recommend",
        )
    )
    return has_language and asks_previous_course and not asks_for_any_course

def answer_course_language(session_id: str) -> Optional[tuple[str, Optional[UserProfile], dict]]:
    course_id = last_recommendations.get(session_id)
    if not course_id:
        return None

    course = orchestrator.course_map.get(course_id)
    if not course:
        return None

    language = course.language or "Chinese"
    response = f"[{course.id}] {course.name} is taught in {language}."
    profile = user_sessions.get(session_id)
    details = {
        "full_output": response,
        "eligible": [],
        "locked": [],
        "query_rag": [],
        "retrieval_methods": {},
        "verdict": {"best_course_id": course.id},
        "score_explanation": SCORE_EXPLANATION,
    }
    return response, profile, details

def run_chat_message(session_id: str, message: str) -> tuple[str, Optional[UserProfile], dict]:
    if is_course_language_question(message):
        language_answer = answer_course_language(session_id)
        if language_answer:
            user_output, profile, details = language_answer
            conversation_logs.append({
                "session_id": session_id,
                "user_message": message,
                "bot_response": user_output,
                "full_output": details.get("full_output"),
                "eligible": [],
                "locked": [],
                "query_rag": [],
                "retrieval_methods": {},
                "verdict": details.get("verdict", {}),
                "response_evaluation": details.get("response_evaluation", {}),
                "score_explanation": details.get("score_explanation", SCORE_EXPLANATION),
                "profile": serialize_profile(profile),
            })
            return language_answer

    profile = user_sessions.get(session_id)
    user_output, new_profile, details = orchestrator.run_user(message, profile=profile)
    if new_profile:
        user_sessions[session_id] = new_profile

    best_course_id = (details.get("verdict") or {}).get("best_course_id")
    if best_course_id:
        last_recommendations[session_id] = best_course_id

    conversation_logs.append({
        "session_id": session_id,
        "user_message": message,
        "bot_response": user_output,
        "full_output": details.get("full_output"),
        "eligible": details.get("eligible", []),
        "locked": details.get("locked", []),
        "query_rag": details.get("query_rag", []),
        "retrieval_methods": details.get("retrieval_methods", {}),
        "verdict": details.get("verdict", {}),
        "response_evaluation": details.get("response_evaluation", {}),
        "score_explanation": details.get("score_explanation", SCORE_EXPLANATION),
        "profile": serialize_profile(new_profile),
    })
    return user_output, new_profile, details

@app.get("/", response_class=HTMLResponse)
async def get_user_interface():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/admin", response_class=HTMLResponse)
async def get_admin_interface():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(request: ChatRequest, req: Request, response: Response):
    session_id = req.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)

    user_output, new_profile, _ = run_chat_message(session_id, request.message)

    return {"response": user_output, "profile": serialize_profile(new_profile)}

@app.websocket("/ws/chat")
async def chat_realtime(websocket: WebSocket):
    await websocket.accept()
    session_id = (
        websocket.query_params.get("session_id")
        or websocket.cookies.get("session_id")
        or str(uuid.uuid4())
    )
    await websocket.send_json({
        "type": "session",
        "session_id": session_id,
        "profile": serialize_profile(user_sessions.get(session_id)),
    })

    try:
        while True:
            payload = await websocket.receive_json()
            message = str(payload.get("message", "")).strip()
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "message": "Please enter a course advising message.",
                })
                continue

            await websocket.send_json({"type": "status", "message": "Analyzing profile and retrieval signals"})
            user_output, new_profile, details = run_chat_message(session_id, message)
            await websocket.send_json({
                "type": "response",
                "response": user_output,
                "profile": serialize_profile(new_profile),
                "verdict": details.get("verdict", {}),
                "eligible": details.get("eligible", []),
                "locked": details.get("locked", []),
                "response_evaluation": details.get("response_evaluation", {}),
            })
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        return


@app.get("/profile")
async def get_profile(req: Request):
    session_id = req.cookies.get("session_id")
    if not session_id or session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="No profile found")
    return user_sessions[session_id].__dict__

@app.post("/admin/login")
async def admin_login(request: LoginRequest, response: Response):
    if request.password == ADMIN_PASSWORD:
        create_admin_session(response)
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid password")

@app.get("/admin/status")
async def admin_status(req: Request):
    return {
        "authenticated": is_admin_request(req),
        "bypass_available": ADMIN_BYPASS_ENABLED,
    }

@app.post("/admin/dev-login")
async def admin_dev_login(response: Response):
    if not ADMIN_BYPASS_ENABLED:
        raise HTTPException(status_code=403, detail="Admin bypass is disabled")
    create_admin_session(response)
    return {"success": True}

@app.post("/admin/logout")
async def admin_logout(req: Request, response: Response):
    token = req.cookies.get("admin_session")
    if token:
        admin_sessions.discard(token)
    response.delete_cookie(key="admin_session")
    response.delete_cookie(key="admin")
    return {"success": True}

@app.post("/admin/add_course")
async def add_course(request: AddCourseRequest, req: Request):
    if not is_admin_request(req):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Validate course data
    required_fields = [
        "id", "name", "credits", "semester", "schedule", "instructor",
        "prerequisites", "description", "department", "language", "degree",
    ]
    course_data = request.course
    missing = [field for field in required_fields if field not in course_data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing)}"
        )

    if not isinstance(course_data["id"], str) or not course_data["id"].strip():
        raise HTTPException(status_code=400, detail="Course ID must be a non-empty string")
    if not isinstance(course_data["name"], str) or not course_data["name"].strip():
        raise HTTPException(status_code=400, detail="Course name must be a non-empty string")
    if not isinstance(course_data["semester"], str) or not course_data["semester"].strip():
        raise HTTPException(status_code=400, detail="Semester must be a non-empty string")
    if not isinstance(course_data["schedule"], str) or not course_data["schedule"].strip():
        raise HTTPException(status_code=400, detail="Schedule must be a non-empty string")
    if not isinstance(course_data["instructor"], str) or not course_data["instructor"].strip():
        raise HTTPException(status_code=400, detail="Instructor must be a non-empty string")
    if not isinstance(course_data["description"], str) or not course_data["description"].strip():
        raise HTTPException(status_code=400, detail="Description must be a non-empty string")
    if not isinstance(course_data["department"], str) or not course_data["department"].strip():
        raise HTTPException(status_code=400, detail="Department must be a non-empty string")

    if not isinstance(course_data["credits"], int) or course_data["credits"] <= 0:
        raise HTTPException(status_code=400, detail="Credits must be a positive integer")

    if not isinstance(course_data["prerequisites"], list) or not all(isinstance(item, str) for item in course_data["prerequisites"]):
        raise HTTPException(status_code=400, detail="Prerequisites must be a list of course IDs")

    if course_data["id"] in VALID_COURSE_IDS:
        raise HTTPException(status_code=400, detail="Course ID already exists")

    language = str(course_data["language"]).strip().title()
    if language not in {"Chinese", "English"}:
        raise HTTPException(status_code=400, detail="Language must be either Chinese or English")

    degree = str(course_data["degree"]).strip().lower()
    if degree not in {"undergrad", "master", "phd"}:
        raise HTTPException(status_code=400, detail="Degree must be one of undergrad, master, phd")

    invalid_prereqs = [p for p in course_data["prerequisites"] if p not in VALID_COURSE_IDS]
    if invalid_prereqs:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prerequisite course IDs: {', '.join(invalid_prereqs)}"
        )

    course_data["language"] = language
    course_data["degree"] = degree

    RAW_COURSES.append(course_data)
    VALID_COURSE_IDS.add(course_data["id"])
    # Reinitialize orchestrator with new courses
    global orchestrator
    orchestrator = CourseFinderOrchestrator()
    return {"success": True}

@app.post("/admin/update_data")
async def update_data(req: Request):
    if not is_admin_request(req):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Rerun RAG: reinitialize agents with updated data
    global orchestrator
    orchestrator = CourseFinderOrchestrator()
    return {"success": True}

@app.get("/admin/logs")
async def get_logs(req: Request):
    if not is_admin_request(req):
        raise HTTPException(status_code=403, detail="Not authorized")
    return conversation_logs

@app.get("/admin/benchmark")
async def get_benchmark(req: Request):
    if not is_admin_request(req):
        raise HTTPException(status_code=403, detail="Not authorized")
    return RetrievalBenchmarkAgent(orchestrator).run()

@app.get("/admin/courses")
async def get_courses(req: Request):
    if not is_admin_request(req):
        raise HTTPException(status_code=403, detail="Not authorized")
    return [c.__dict__ for c in orchestrator.bm25_agent.courses]

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
