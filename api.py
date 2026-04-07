from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import json
import os
from typing import Optional, Dict, Any
from agents.OrchestratorAgent import CourseFinderOrchestrator
from models.UserProfile import RAW_COURSES, UserProfile, VALID_COURSE_IDS
from models.Course import Course

app = FastAPI()

# Global orchestrator
orchestrator = CourseFinderOrchestrator()

# In-memory storage for simplicity (use database in production)
user_sessions: Dict[str, UserProfile] = {}
conversation_logs: list = []

# Admin password (hardcoded for demo)
ADMIN_PASSWORD = "admin123"

class ChatRequest(BaseModel):
    message: str

class AddCourseRequest(BaseModel):
    course: Dict[str, Any]

class LoginRequest(BaseModel):
    password: str

@app.get("/", response_class=HTMLResponse)
async def get_user_interface():
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/admin", response_class=HTMLResponse)
async def get_admin_interface():
    with open("static/admin.html", "r") as f:
        return f.read()

@app.post("/chat")
async def chat(request: ChatRequest, req: Request, response: Response):
    session_id = req.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)

    profile = user_sessions.get(session_id)
    user_output, new_profile, details = orchestrator.run_user(request.message, profile=profile)
    if new_profile:
        user_sessions[session_id] = new_profile

    # Log conversation for admin review
    conversation_logs.append({
        "session_id": session_id,
        "user_message": request.message,
        "bot_response": user_output,
        "full_output": details.get("full_output"),
        "eligible": details.get("eligible", []),
        "locked": details.get("locked", []),
        "verdict": details.get("verdict", {}),
        "profile": new_profile.__dict__ if new_profile else None,
    })

    return {"response": user_output, "profile": new_profile.__dict__ if new_profile else None}

@app.get("/profile")
async def get_profile(req: Request):
    session_id = req.cookies.get("session_id")
    if not session_id or session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="No profile found")
    return user_sessions[session_id].__dict__

@app.post("/admin/login")
async def admin_login(request: LoginRequest, response: Response):
    if request.password == ADMIN_PASSWORD:
        response.set_cookie(key="admin", value="true")
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid password")

@app.post("/admin/add_course")
async def add_course(request: AddCourseRequest, req: Request):
    if req.cookies.get("admin") != "true":
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
    if req.cookies.get("admin") != "true":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Rerun RAG: reinitialize agents with updated data
    global orchestrator
    orchestrator = CourseFinderOrchestrator()
    return {"success": True}

@app.get("/admin/logs")
async def get_logs(req: Request):
    if req.cookies.get("admin") != "true":
        raise HTTPException(status_code=403, detail="Not authorized")
    return conversation_logs

@app.get("/admin/courses")
async def get_courses(req: Request):
    if req.cookies.get("admin") != "true":
        raise HTTPException(status_code=403, detail="Not authorized")
    return [c.__dict__ for c in orchestrator.bm25_agent.courses]

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")