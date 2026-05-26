# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional

from agents.IntakeAgent import IntakeAgent
from agents.BM25 import BM25Agent
from agents.FusionAgent import FusionAgent
from agents.JudgeAgent import JudgeAgent
from agents.QueryRAGAgent import QueryRAGAgent
from agents.ResponseAgent import ResponseAgent
from agents.VectorAgent import VectorAgent
from config.main import GROQ_DEFAULT_MODEL
from keywords.OffTopicResponse import OFF_TOPIC_RESPONSE
from models.Course import Course
from models.RetrievalResult import RetrievalResult
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

    def __init__(self, model: str = GROQ_DEFAULT_MODEL, provider: str = "groq"):
        self.model    = model
        self.provider = provider
        courses = [Course(**c) for c in RAW_COURSES]
        self.course_map = {c.id: c for c in courses}

        self.intake_agent = IntakeAgent(model=model, provider=provider)
        self.judge_agent  = JudgeAgent(model=model, provider=provider)
        self.bm25_agent     = BM25Agent(courses)
        self.vector_agent   = VectorAgent(courses)
        self.query_rag_agent = QueryRAGAgent(courses, dense_agent=self.vector_agent)
        self.fusion_agent   = FusionAgent()
        self.response_agent = ResponseAgent()

    def _append_prerequisite_path(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        seen = {r.course.id for r in results}
        expanded = set()
        queue = [pid for r in results for pid in r.course.prerequisites]
        bridged = list(results)

        while queue:
            course_id = queue.pop(0)
            course = self.course_map.get(course_id)
            if not course:
                continue

            if course_id not in seen:
                seen.add(course_id)
                bridged.append(
                    RetrievalResult(
                        course=course,
                        score=max(0.0001, 0.001 / (len(bridged) + 1)),
                        source="prerequisite_path",
                    )
                )

            if course_id not in expanded:
                expanded.add(course_id)
                queue.extend(course.prerequisites)

        return bridged

    def _run_pipeline(
        self,
        raw_input: str,
        profile: Optional[UserProfile] = None,
    ) -> tuple[str, str, UserProfile | None, dict]:
            # 1. Update/Create Student Profile
            profile = self.intake_agent.process(raw_input, model=self.model, existing_profile=profile)
            if profile is None:
                return OFF_TOPIC_RESPONSE, OFF_TOPIC_RESPONSE, None, {}

            # 2. Parallel Retrieval
            bm25_results   = self.bm25_agent.process(profile, top_k=6)
            vector_results = self.vector_agent.process(profile, top_k=6)
            query_rankings = self.query_rag_agent.process(profile, top_k=6)
            retrieval_rankings = {
                "bm25_agent": bm25_results,
                "vector_agent": vector_results,
                **query_rankings,
            }
            
            # 3. Fusion & Logical Filtering
            # This should return two lists: 
            # 'eligible' (Uncompleted + Prereqs met) and 'locked' (Uncompleted + Prereqs missing)
            query_fused = self.query_rag_agent.fuse(retrieval_rankings)
            query_fused = self._append_prerequisite_path(query_fused)
            eligible, locked = self.fusion_agent.filter_results(query_fused, profile)

            # 4. The Judge's Decision
            # CRITICAL: We only pass 'eligible' courses to the judge.
            # This prevents recommending MATH2002 if the student already passed it.
            if not eligible:
                verdict = None 
            else:
                verdict = self.judge_agent.process(profile, eligible)

            # 5. Final Formatting
            full_output = self.response_agent.process(
                profile=profile,
                eligible_results=eligible,
                locked_results=locked,
                bm25_results=bm25_results,
                vector_results=vector_results,
                verdict=verdict,
                course_map=self.course_map,
            )
            user_output = self.response_agent.minimal_response(
                verdict,
                self.course_map,
                locked_results=locked,
                profile=profile,
            )
            details = {
                "eligible": [
                    {
                        "id": r.course.id,
                        "name": r.course.name,
                        "score": r.score,
                        "source": r.source,
                    }
                    for r in eligible
                ],
                "locked": [
                    {
                        "id": r.course.id,
                        "name": r.course.name,
                        "score": r.score,
                        "source": r.source,
                        "missing_prereqs": r.missing_prereqs,
                        "filter_reason": r.filter_reason,
                    }
                    for r in locked
                ],
                "verdict": {
                    "best_course_id": verdict.best_course_id if verdict else None,
                    "runner_up_id": verdict.runner_up_id if verdict else None,
                    "reasoning": verdict.reasoning if verdict else None,
                    "confidence": verdict.confidence if verdict else None,
                },
                "full_output": full_output,
                "query_rag": [
                    {
                        "id": r.course.id,
                        "name": r.course.name,
                        "score": r.score,
                        "source": r.source,
                    }
                    for r in query_fused
                ],
                "retrieval_methods": {
                    name: [
                        {
                            "id": r.course.id,
                            "name": r.course.name,
                            "score": r.score,
                            "source": r.source,
                        }
                        for r in results
                    ]
                    for name, results in retrieval_rankings.items()
                },
            }
            return full_output, user_output, profile, details

    def run(self, raw_input: str, profile: Optional[UserProfile] = None) -> tuple[str, UserProfile | None]:
            full_output, _, profile, _ = self._run_pipeline(raw_input, profile)
            return full_output, profile

    def run_user(
        self,
        raw_input: str,
        profile: Optional[UserProfile] = None,
    ) -> tuple[str, UserProfile | None, dict]:
            _, user_output, profile, details = self._run_pipeline(raw_input, profile)
            return user_output, profile, details
