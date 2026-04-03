# Multi-Agent Course Finder v2 — NCU Midterm

### BM25 + Llama3 Vectors + Function Calling (Intake & Judge Agents)

---

## What's New in v2

| Feature                  | v1                       | v2                                                                            |
| ------------------------ | ------------------------ | ----------------------------------------------------------------------------- |
| User input parsing       | Simple string validation | **IntakeAgent** — Llama3 function call extracts structured `UserProfile`      |
| Final answer             | Full ranked list         | **JudgeAgent** — Llama3 function call picks single best course with reasoning |
| Query used for retrieval | Raw user string          | Synthesised `search_query` from profile                                       |
| Output                   | Ranked list only         | Ranked list + judge verdict + confidence level                                |

---

## Architecture

```
User free-text input
        │
        ▼
┌─────────────────────────────────────────────────┐
│  IntakeAgent  (Ollama function call)            │
│                                                 │
│  Tool: extract_user_profile                     │
│  Output: UserProfile {                          │
│    academic_year, completed_courses,            │
│    goals, constraints, search_query             │
│  }                                              │
└──────────────────┬──────────────────────────────┘
                   │  profile.search_query
          ┌────────┴────────┐
          ▼                 ▼
   ┌────────────┐    ┌─────────────────┐
   │ BM25Agent  │    │  VectorAgent    │
   │ (keyword)  │    │  Llama3 + Chroma│
   └─────┬──────┘    └────────┬────────┘
         │                    │
         └──────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │  FusionAgent     │
           │  RRF (k=60)      │
           └────────┬─────────┘
                    │  ranked candidates
                    ▼
┌─────────────────────────────────────────────────┐
│  JudgeAgent  (Ollama function call)             │
│                                                 │
│  Tool: select_best_course                       │
│  Input: UserProfile + RRF-ranked candidates     │
│  Output: JudgeVerdict {                         │
│    best_course_id, runner_up_id,                │
│    reasoning, confidence                        │
│  }                                              │
└──────────────────┬──────────────────────────────┘
                   ▼
          ┌─────────────────┐
          │  ResponseAgent  │
          │  Final output   │
          └─────────────────┘
```

---

## Agents

| #   | Agent             | Role                                                 | Ollama call?                   |
| --- | ----------------- | ---------------------------------------------------- | ------------------------------ |
| 1   | **IntakeAgent**   | Structured extraction of user profile from free text | ✅ `extract_user_profile` tool |
| 2   | **BM25Agent**     | Keyword retrieval with BM25Okapi                     | —                              |
| 3   | **VectorAgent**   | Semantic retrieval with Llama3 embeddings + ChromaDB | ✅ `ollama.embeddings`         |
| 4   | **FusionAgent**   | Reciprocal Rank Fusion (RRF, k=60)                   | —                              |
| 5   | **JudgeAgent**    | Picks single best course with reasoning              | ✅ `select_best_course` tool   |
| 6   | **ResponseAgent** | Formats all results into final output                | —                              |

---

## Function Call Tool Schemas

### `extract_user_profile` (IntakeAgent)

```json
{
  "academic_year": 1,
  "completed_courses": ["CSIE1001"],
  "goals": ["learn algorithms", "prepare for ML"],
  "constraints": ["prefer morning classes"],
  "search_query": "algorithms data structures foundations for machine learning"
}
```

### `select_best_course` (JudgeAgent)

```json
{
  "best_course_id": "CSIE1001",
  "runner_up_id": "CSIE1002",
  "reasoning": "CSIE1001 directly matches the student's goal of learning programming from scratch. As a first-year student with no prerequisites met, this is the only appropriate starting point.",
  "confidence": "high"
}
```

---

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ollama + Llama3
brew install ollama       # macOS
curl -fsSL https://ollama.com/install.sh | sh  # Linux

ollama pull llama3
ollama serve
```

---

## Running

```bash
# Interactive REPL
python main.py

# Single query
python main.py -q "I'm a sophomore. I finished CSIE1001 and want to study data structures"
python main.py -q "complete beginner, interested in math and logic"

# Use a different model
python main.py --model llama3.1 -q "I want to learn about trees and graphs"
```

### Example output

```
=================================================================
  NCU Course Finder v2 — Multi-Agent RAG + Judge
=================================================================

── Student Profile ───────────────────────────────────────────
Year       : 2
Completed  : CSIE1001
Goals      : study data structures; algorithm complexity
Constraints: none
Query      : data structures linked lists trees graphs algorithm complexity

── Retrieval Pipeline ────────────────────────────────────────
  Method : BM25 (keyword) + Llama3 (semantic) → RRF fusion

  #1  CSIE2001   RRF=0.032258  BM25 rank=1  Vec rank=1  ◄ JUDGE PICK
  #2  CSIE1002   RRF=0.016260  BM25 rank=2  Vec rank=2
  #3  CSIE1001   RRF=0.010989  BM25 rank=3  Vec rank=3

── Judge Verdict ─────────────────────────────────────────────
  Best Course   : [CSIE2001] Data Structures
  Confidence    : HIGH  ★★★
  Semester      : Fall / Spring
  Schedule      : Monday 13:00–15:00, Wednesday 13:00–14:00
  Instructor    : Prof. Wang Da-Ming
  Prerequisites : CSIE1001

  Why this course?
    CSIE2001 directly matches the student's stated interest
    in data structures and algorithm complexity. The student
    has already completed the only prerequisite (CSIE1001),
    making this the natural next step in their CS curriculum.

  Runner-up     : [CSIE1002] Discrete Mathematics
=================================================================
```

---

## Tests

```bash
pip install pytest
python -m pytest test_course_finder.py -v
```

46 tests covering: tool schemas, IntakeAgent (function call + fallback), JudgeAgent (function call + fallback + ID validation), BM25Agent, FusionAgent, ResponseAgent, and full pipeline integration with mocked Ollama.

---

## Offline Fallback

Both function-calling agents degrade gracefully if Ollama is unavailable:

- **IntakeAgent** → heuristic regex extraction (year keywords, course ID detection, goal phrases)
- **VectorAgent** → in-memory TF-IDF cosine similarity
- **JudgeAgent** → returns RRF rank #1 with `confidence=low`

---

## Files

```
course_finder_v2/
├── main.py           # All 6 agents + orchestrator
├── tests.py          # 46 unit + integration tests
├── requirements.txt
└── README.md
```
