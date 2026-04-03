# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional

from agents.IntakeAgent import IntakeAgent
from agents.BM25 import BM25Agent
from agents.FusionAgent import FusionAgent
from agents.JudgeAgent import JudgeAgent
from agents.ResponseAgent import ResponseAgent
from agents.VectorAgent import VectorAgent
from config.main import GROQ_DEFAULT_MODEL
from keywords.OffTopicResponse import OFF_TOPIC_RESPONSE
from models.Course import Course
from models.UserProfile import RAW_COURSES, UserProfile

class CourseFinderOrchestrator:
    """
    Wires all agents together and routes messages through the pipeline.

    Pipeline:
      IntakeAgent  (Groq function call → UserProfile)
          │
          ├──► BM25Agent   (keyword retrieval)
          └──► VectorAgent (Sentence-Transformers + ChromaDB)
                   │
               FusionAgent (RRF)
                   │
               JudgeAgent  (Groq function call → single best pick)
                   │
               ResponseAgent (formatted output)
    """

    def __init__(self, model: str = GROQ_DEFAULT_MODEL):
        self.model = model
        courses = [Course(**c) for c in RAW_COURSES]
        self.course_map = {c.id: c for c in courses}

        self.intake_agent   = IntakeAgent()
        self.bm25_agent     = BM25Agent(courses)
        self.vector_agent   = VectorAgent(courses)
        self.fusion_agent   = FusionAgent()
        self.judge_agent    = JudgeAgent()
        self.response_agent = ResponseAgent()

    def run(self, raw_input: str, profile: Optional[UserProfile] = None) -> tuple[str, UserProfile | None]:
            # 1. Update/Create Student Profile
            profile = self.intake_agent.process(raw_input, model=self.model, existing_profile=profile)
            if profile is None:
                return OFF_TOPIC_RESPONSE, None

            # 2. Parallel Retrieval
            bm25_results   = self.bm25_agent.process(profile, top_k=6)
            vector_results = self.vector_agent.process(profile, top_k=6)
            
            # 3. Fusion & Logical Filtering
            # This should return two lists: 
            # 'eligible' (Uncompleted + Prereqs met) and 'locked' (Uncompleted + Prereqs missing)
            eligible, locked = self.fusion_agent.process(bm25_results, vector_results, profile)

            # 4. The Judge's Decision
            # CRITICAL: We only pass 'eligible' courses to the judge.
            # This prevents recommending MATH2002 if the student already passed it.
            if not eligible:
                # Fallback if no courses are eligible based on the current search
                verdict = None 
            else:
                verdict = self.judge_agent.process(profile, eligible, model=self.model)

            # 5. Final Formatting
            output = self.response_agent.process(
                profile=profile,
                eligible_results=eligible,
                locked_results=locked,
                bm25_results=bm25_results,
                vector_results=vector_results,
                verdict=verdict,
                course_map=self.course_map,
            )
            return output, profile
