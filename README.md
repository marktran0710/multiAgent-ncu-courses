# Multi-Agent Course Finder v2 — NCU CSIE Midterm

### BM25 + Sentence-Transformers + Groq Function Calling (Intake & Judge Agents)

---

## What's New in v2

| Feature                  | v1                       | v2                                                                          |
| ------------------------ | ------------------------ | --------------------------------------------------------------------------- |
| User input parsing       | Simple string validation | **IntakeAgent** — Groq function call extracts structured `UserProfile`      |
| Final answer             | Full ranked list         | **JudgeAgent** — Groq function call picks single best course with reasoning |
| Query used for retrieval | Raw user string          | Synthesised `search_query` from profile                                     |
| Output                   | Ranked list only         | Ranked list + judge verdict + confidence level                              |
| LLM backend              | Ollama (local)           | **Groq API** (`llama-3.3-70b-versatile`)                                    |
| Student support          | Undergrad only           | **Undergrad + Master's + PhD** with `degree_level` field                    |
| Embeddings               | Ollama Llama3            | **Sentence-Transformers** (`all-MiniLM-L6-v2`)                              |
| Session memory           | Stateless (single turn)  | **Multi-turn** — `UserProfile.update()` merges follow-up input across turns |
| Course catalogue         | 3 undergrad courses      | **6 courses** — 3 undergrad + 2 Master's + 1 PhD                            |

---

## Architecture

```
User free-text input  (single-turn or multi-turn follow-up)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  IntakeAgent  (Groq function call)                      │
│                                                         │
│  Tool: extract_user_profile                             │
│  First turn  → builds fresh UserProfile                 │
│  Follow-up   → merges changes via UserProfile.update()  │
│                                                         │
│  Output: UserProfile {                                  │
│    degree_level, academic_year,                         │
│    completed_courses, goals,                            │
│    constraints, search_query                            │
│  }                                                      │
└──────────────────┬──────────────────────────────────────┘
                   │  profile.search_query
          ┌────────┴────────┐
          ▼                 ▼
   ┌────────────┐    ┌──────────────────────────┐
   │ BM25Agent  │    │  VectorAgent             │
   │ (keyword)  │    │  Sentence-Transformers   │
   │            │    │  + ChromaDB              │
   └─────┬──────┘    └────────┬─────────────────┘
         │                    │
         └──────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │  FusionAgent     │
           │  RRF (k=60)      │
           └────────┬─────────┘
                    │  ranked candidates
                    ▼
┌─────────────────────────────────────────────────────────┐
│  JudgeAgent  (Groq function call)                       │
│                                                         │
│  Tool: select_best_course                               │
│  Input: UserProfile + RRF-ranked candidates             │
│  Steers toward correct level:                           │
│    Undergrad → CSIE1xxx/2xxx                            │
│    Master's  → CSIE5xxx                                 │
│    PhD       → CSIE5xxx/6xxx                            │
│                                                         │
│  Output: JudgeVerdict {                                 │
│    best_course_id, runner_up_id,                        │
│    reasoning, confidence                                │
│  }                                                      │
└──────────────────┬──────────────────────────────────────┘
                   ▼
          ┌─────────────────┐
          │  ResponseAgent  │
          │  Final output   │
          └─────────────────┘
```

---

## Agents

| #   | Agent             | Role                                                       | Groq call?                     |
| --- | ----------------- | ---------------------------------------------------------- | ------------------------------ |
| 1   | **IntakeAgent**   | Extracts / updates structured `UserProfile` from free text | ✅ `extract_user_profile` tool |
| 2   | **BM25Agent**     | Keyword retrieval with BM25Okapi (top-5)                   | —                              |
| 3   | **VectorAgent**   | Semantic retrieval with Sentence-Transformers + ChromaDB   | —                              |
| 4   | **FusionAgent**   | Reciprocal Rank Fusion (RRF, k=60)                         | —                              |
| 5   | **JudgeAgent**    | Picks single best course with degree-aware reasoning       | ✅ `select_best_course` tool   |
| 6   | **ResponseAgent** | Formats all results into final output                      | —                              |

---

## Graduate Student Support

Academic year encoding spans both undergrad and graduate programmes:

| Degree Level  | `degree_level` | `academic_year` range |
| ------------- | -------------- | --------------------- |
| Undergraduate | `"undergrad"`  | 1 – 4                 |
| Master's      | `"master"`     | 5 – 6                 |
| PhD           | `"phd"`        | 7 – 10                |

The IntakeAgent infers degree level from natural language:

- `"grad student"`, `"Master's"`, `"MSc"`, `"thesis"` → `master`
- `"PhD"`, `"doctoral"`, `"dissertation"`, `"candidacy"` → `phd`
- No hint / `"freshman"` / `"new to programming"` → `undergrad`

The JudgeAgent steers recommendations toward the appropriate course level.

---

## Course Catalogue

| ID       | Name                                 | Level     | Semester      |
| -------- | ------------------------------------ | --------- | ------------- |
| CSIE1001 | Introduction to Programming          | Undergrad | Fall / Spring |
| CSIE1002 | Discrete Mathematics                 | Undergrad | Fall          |
| CSIE2001 | Data Structures                      | Undergrad | Fall / Spring |
| CSIE5001 | Advanced Machine Learning            | Graduate  | Fall          |
| CSIE5002 | Distributed Systems                  | Graduate  | Spring        |
| CSIE6001 | Research Methods in Computer Science | Graduate  | Fall / Spring |

---

## Multi-Turn Profile Updates

`UserProfile.update()` merges follow-up input into the existing profile without overwriting prior information:

- **Lists** (`completed_courses`, `goals`, `constraints`) are **appended and deduplicated** — old data is never lost.
- **Scalar fields** (`degree_level`, `academic_year`, `search_query`) are only overwritten if the new message explicitly changes them.
- **Groq failure on update** — the existing profile is returned unchanged rather than resetting to defaults.

The orchestrator's `run()` returns `(output, profile)` so the REPL carries the profile across turns:

```python
profile = None
while True:
    raw = input("You: ").strip()
    output, profile = orchestrator.run(raw, profile=profile)
    print(output)
```

---

## Function Call Tool Schemas

### `extract_user_profile` (IntakeAgent)

```json
{
  "degree_level": "master",
  "academic_year": 5,
  "completed_courses": ["CSIE2001"],
  "goals": ["deep learning", "NLP research"],
  "constraints": ["no early morning classes"],
  "search_query": "advanced machine learning deep learning NLP transformers graduate research"
}
```

### `select_best_course` (JudgeAgent)

```json
{
  "best_course_id": "CSIE5001",
  "runner_up_id": "CSIE5002",
  "reasoning": "CSIE5001 directly matches the student's NLP and deep learning research goals. As a first-year Master's student who has completed Data Structures, they meet the prerequisite and are ready for graduate-level ML content.",
  "confidence": "high"
}
```

---

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt
```

Set your Groq API key in a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

---

## Running

```bash
# Interactive multi-turn REPL
python main.py

# Single query
python main.py -q "I'm a first-year Master's student interested in machine learning"
python main.py -q "PhD student looking for research methods and academic writing"
python main.py -q "sophomore, finished CSIE1001, want to study data structures"

# Use a different Groq model
python main.py --model llama-3.3-70b-versatile -q "I want to learn about graphs and algorithms"
```

### Example output (graduate student)

```
=================================================================
  NCU Course Finder v2 — Multi-Agent RAG + Judge
=================================================================

── Student Profile ───────────────────────────────────────────
Degree     : Master's
Year       : 5
Completed  : CSIE2001
Goals      : deep learning; NLP research
Constraints: none
Query      : advanced machine learning deep learning NLP transformers

── Retrieval Pipeline ────────────────────────────────────────
  Method : BM25 (keyword) + Sentence-Transformers (semantic) → RRF fusion

  #1  CSIE5001   [graduate]   RRF=0.032258  BM25 rank=1  Vec rank=1  ◄ JUDGE PICK
  #2  CSIE6001   [graduate]   RRF=0.016260  BM25 rank=2  Vec rank=2
  #3  CSIE5002   [graduate]   RRF=0.010989  BM25 rank=3  Vec rank=3

── Judge Verdict ─────────────────────────────────────────────
  Best Course   : [CSIE5001] Advanced Machine Learning  [GRADUATE]
  Confidence    : HIGH  ★★★
  Semester      : Fall
  Schedule      : Wednesday 14:00–17:00
  Instructor    : Prof. Huang Zhi-Yuan
  Prerequisites : CSIE2001

  Why this course?
    CSIE5001 directly aligns with the student's deep learning
    and NLP research goals. As a first-year Master's student
    who has completed Data Structures, they meet the
    prerequisite and are ready for graduate-level ML content.

  Runner-up     : [CSIE6001] Research Methods in CS  [GRADUATE]
=================================================================
```

### Example multi-turn session

```
You: I'm a grad student interested in machine learning
→ [Degree: Master's | Goals: machine learning]

You: actually I'm doing a PhD, and I also need help with academic writing
→ [Degree updated to PhD | Goals: machine learning; academic writing]

You: I've already completed CSIE2001
→ [Completed: CSIE2001 added — prior goals preserved]
```

---

## Offline Fallback

All agents degrade gracefully when Groq or Sentence-Transformers is unavailable:

| Agent                         | Fallback behaviour                                                              |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **IntakeAgent**               | Heuristic regex — detects year keywords, degree level, course IDs, goal phrases |
| **VectorAgent**               | In-memory TF-IDF cosine similarity                                              |
| **JudgeAgent**                | Returns RRF rank #1 with `confidence=low`                                       |
| **IntakeAgent (update mode)** | Returns existing profile unchanged                                              |

---

## Tests

```bash
pip install pytest
python -m pytest tests.py -v
```

46 tests covering: tool schemas, IntakeAgent (function call + fallback + update/merge), JudgeAgent (function call + fallback + ID validation), BM25Agent, FusionAgent, ResponseAgent, graduate student routing, and full pipeline integration with mocked Groq.

---

## Files

```
course_finder_v2/
├── main.py           # All 6 agents + orchestrator
├── tests.py          # 46 unit + integration tests
├── requirements.txt
└── README.md
```

## Requirements

```
groq
rank-bm25
chromadb
sentence-transformers
python-dotenv
pytest
```
