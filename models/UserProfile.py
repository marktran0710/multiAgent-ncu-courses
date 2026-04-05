

from dataclasses import dataclass

RAW_COURSES = [
    {
        "id": "CSIE1001",
        "name": "Introduction to Programming",
        "credits": 3,
        "semester": "Fall / Spring",
        "schedule": "Monday 10:00–12:00, Thursday 10:00–11:00",
        "instructor": "Prof. Chen Wei",
        "prerequisites": [],
        "description": (
            "Fundamental programming concepts using Python. Topics include variables, "
            "control flow, functions, recursion, and basic data structures. "
            "Suitable for students with no prior programming experience."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE1002",
        "name": "Discrete Mathematics",
        "credits": 3,
        "semester": "Fall",
        "schedule": "Tuesday 09:00–12:00",
        "instructor": "Prof. Lin Mei-Hua",
        "prerequisites": [],
        "description": (
            "Set theory, logic, relations, functions, graph theory, combinatorics, "
            "and proof techniques. Essential foundation for upper-level CS courses."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE2001",
        "name": "Data Structures",
        "credits": 3,
        "semester": "Fall / Spring",
        "schedule": "Monday 13:00–15:00, Wednesday 13:00–14:00",
        "instructor": "Prof. Wang Da-Ming",
        "prerequisites": ["CSIE1001"],
        "description": (
            "Arrays, linked lists, stacks, queues, trees, heaps, hash tables, and graphs. "
            "Emphasis on algorithm complexity and space-time tradeoffs."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE2002",
        "name": "Computer Organization",
        "credits": 3,
        "semester": "Fall",
        "schedule": "Tuesday 13:00–15:00, Friday 13:00–14:00",
        "instructor": "Prof. Huang Jia-Wei",
        "prerequisites": ["CSIE1001"],
        "description": (
            "Digital logic, CPU design, instruction sets, memory hierarchy, I/O systems. "
            "Includes assembly language programming labs."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE3001",
        "name": "Algorithms",
        "credits": 3,
        "semester": "Fall / Spring",
        "schedule": "Wednesday 10:00–12:00, Friday 10:00–11:00",
        "instructor": "Prof. Chang Shu-Fen",
        "prerequisites": ["CSIE2001", "CSIE1002"],
        "description": (
            "Divide and conquer, dynamic programming, greedy algorithms, graph algorithms, "
            "NP-completeness. Students will analyze and implement classic algorithms."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE3002",
        "name": "Operating Systems",
        "credits": 3,
        "semester": "Spring",
        "schedule": "Monday 10:00–12:00, Wednesday 10:00–11:00",
        "instructor": "Prof. Liu Zhi-Yuan",
        "prerequisites": ["CSIE2001", "CSIE2002"],
        "description": (
            "Process management, scheduling, memory management, file systems, I/O, "
            "concurrency, and synchronization. Kernel programming projects included."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE4001",
        "name": "Machine Learning",
        "credits": 3,
        "semester": "Fall / Spring",
        "schedule": "Tuesday 14:00–17:00",
        "instructor": "Prof. Tsai Mei-Ling",
        "prerequisites": ["CSIE3001", "MATH2001"],
        "description": (
            "Supervised and unsupervised learning, regression, classification, neural networks, "
            "SVMs, clustering, dimensionality reduction. Includes Kaggle competition project."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE4002",
        "name": "Deep Learning",
        "credits": 3,
        "semester": "Spring",
        "schedule": "Thursday 14:00–17:00",
        "instructor": "Prof. Tsai Mei-Ling",
        "prerequisites": ["CSIE4001"],
        "description": (
            "CNNs, RNNs, transformers, generative models, reinforcement learning. "
            "PyTorch-based projects including image classification and NLP tasks."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "MATH2001",
        "name": "Linear Algebra",
        "credits": 3,
        "semester": "Fall / Spring",
        "schedule": "Monday 08:00–10:00, Wednesday 08:00–09:00",
        "instructor": "Prof. Chou Li-Chen",
        "prerequisites": [],
        "description": (
            "Vectors, matrices, linear transformations, eigenvalues, eigenvectors, "
            "singular value decomposition. Essential for machine learning and computer graphics."
        ),
        "department": "Mathematics",
    },
    {
        "id": "MATH2002",
        "name": "Probability and Statistics",
        "credits": 3,
        "semester": "Fall / Spring",
        "schedule": "Tuesday 10:00–12:00, Thursday 10:00–11:00",
        "instructor": "Prof. Wu Chun-Hao",
        "prerequisites": [],
        "description": (
            "Probability theory, random variables, distributions, estimation, hypothesis "
            "testing, regression. Required for data science track students."
        ),
        "department": "Mathematics",
    },
    {
        "id": "CSIE4003",
        "name": "Natural Language Processing",
        "credits": 3,
        "semester": "Fall",
        "schedule": "Wednesday 14:00–17:00",
        "instructor": "Prof. Ko Wen-Jie",
        "prerequisites": ["CSIE4001", "MATH2002"],
        "description": (
            "Tokenization, language models, embeddings, transformers, named entity recognition, "
            "sentiment analysis, machine translation, and QA systems."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE4004",
        "name": "Computer Vision",
        "credits": 3,
        "semester": "Spring",
        "schedule": "Friday 14:00–17:00",
        "instructor": "Prof. Shih Ying-Jui",
        "prerequisites": ["CSIE4001"],
        "description": (
            "Image processing, feature extraction, convolutional networks, object detection, "
            "segmentation, 3D vision, and video understanding."
        ),
        "department": "Computer Science and Information Engineering",
    },
    {
        "id": "CSIE6001",
        "name": "Research Methods in Computer Science",
        "credits": 2,
        "semester": "Fall / Spring",
        "schedule": "Friday 10:00–12:00",
        "instructor": "Prof. Liu Pei-Shan",
        "prerequisites": [],
        "description": (
            "Academic writing, literature review, experimental design, "
            "statistical analysis, and paper publication process. "
            "Mandatory for all PhD students; strongly recommended for Master's students "
            "planning to write a thesis."
        ),
        "department": "Computer Science and Information Engineering",
    },
]

DEGREE_YEAR_RANGES = {
    "undergrad": (1, 4),
    "master":    (5, 6),
    "phd":       (7, 10),
}

VALID_COURSE_IDS = {c["id"] for c in RAW_COURSES}

def degree_from_year(year: int) -> str:
    if year <= 4:
        return "undergrad"
    elif year <= 6:
        return "master"
    return "phd"

@dataclass
class UserProfile:
    """Structured user profile extracted by IntakeAgent via function calling."""
    raw_input: str
    academic_year: int
    degree_level: str          # "undergrad" | "master" | "phd"
    completed_courses: list[str]
    goals: list[str]
    constraints: list[str]
    search_query: str

    def _is_similar_goal(self, new_goal: str, existing_goals: list[str], threshold: float = 0.6) -> bool:
        # ↑ must be indented INSIDE the class — 4 spaces
        STOP_WORDS = {"i", "to", "a", "the", "and", "or", "in", "want", "learn", "study", "take"}

        def key_words(text: str) -> set[str]:
            return {w.rstrip("s") for w in text.lower().split() if w not in STOP_WORDS}

        new_words = key_words(new_goal)
        if not new_words:
            return True

        for g in existing_goals:
            existing_words = key_words(g)
            if not existing_words:
                continue
            intersection = new_words & existing_words
            union        = new_words | existing_words
            jaccard      = len(intersection) / len(union)
            if jaccard >= threshold:
                return True
        return False

    def update(self, new_input: str, args: dict) -> None:
        self.raw_input = new_input

        # ── academic_year + degree_level ─────────────────────────────────
        if "academic_year" in args:
            new_year = max(1, min(10, int(args["academic_year"])))
            # allow upgrade AND downgrade — always trust explicit LLM value
            self.academic_year = new_year
            self.degree_level  = degree_from_year(new_year)  # always re-derive

        # ── completed_courses ─────────────────────────────────────────────
        if "completed_courses" in args:
            incoming = [
                c for c in (args["completed_courses"] or [])
                if c in VALID_COURSE_IDS
                and c not in self.completed_courses
            ]
            self.completed_courses = self.completed_courses + incoming

        # ── goals ─────────────────────────────────────────────────────────
        if "goals" in args:
            new_goals = [
                g.strip() for g in (args["goals"] or [])
                if g.strip()
                and not self._is_similar_goal(g.strip(), self.goals)
            ]
            self.goals = (self.goals + new_goals)[-6:]

        # ── constraints ───────────────────────────────────────────────────
        if "constraints" in args:
            incoming = args["constraints"] or []
            removals = [
                c for c in incoming
                if any(kw in c.lower() for kw in ("no longer", "not anymore", "removed"))
            ]
            additions = [
                c.strip() for c in incoming
                if c.strip()
                and c not in removals
                and not any(c.strip().lower() == x.lower() for x in self.constraints)
            ]
            self.constraints = [
                c for c in self.constraints
                if not any(r.lower() in c.lower() for r in removals)
            ] + additions

        # ── search_query ──────────────────────────────────────────────────
        if "search_query" in args and args["search_query"].strip():
            self.search_query = args["search_query"].strip()
        else:
            self.search_query = self._build_search_query()

    def _build_search_query(self) -> str:
        """Rebuild a fresh search query from current profile state."""
        parts = []
        if self.goals:
            parts.append(" ".join(self.goals[:3]))
        if self.completed_courses:
            parts.append(f"after completing {' '.join(self.completed_courses[-3:])}")
        if self.constraints:
            parts.append(" ".join(self.constraints[:2]))
        return " ".join(parts) if parts else self.raw_input

    def is_complete(self) -> bool:
        """Check if profile has enough info for a meaningful recommendation."""
        return bool(self.goals or self.search_query)

    def describe(self) -> str:
        degree_label = {
            "undergrad": "Undergraduate",
            "master":    "Master's",
            "phd":       "PhD",
        }.get(self.degree_level, self.degree_level)

        completed    = ", ".join(self.completed_courses) if self.completed_courses else "none"
        goals        = "; ".join(self.goals)             if self.goals             else "not specified"
        constraints  = "; ".join(self.constraints)       if self.constraints       else "none"

        return (
            f"Degree     : {degree_label}\n"
            f"Year       : {self.academic_year}\n"
            f"Completed  : {completed}\n"
            f"Goals      : {goals}\n"
            f"Constraints: {constraints}\n"
            f"Query      : {self.search_query}"
        )

    def __repr__(self) -> str:
        return (
            f"UserProfile(year={self.academic_year}, "
            f"degree={self.degree_level}, "
            f"completed={self.completed_courses}, "
            f"goals={self.goals})"
        )
