

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
        "language": "English",
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
        "language": "English",
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
        "language": "English",
        "degree": "master",
    },
]

def _course(
    course_id: str,
    name: str,
    credits: int,
    semester: str,
    schedule: str,
    instructor: str,
    prerequisites: list[str],
    description: str,
    department: str,
    language: str = "Chinese",
    degree: str = "undergrad",
) -> dict:
    return {
        "id": course_id,
        "name": name,
        "credits": credits,
        "semester": semester,
        "schedule": schedule,
        "instructor": instructor,
        "prerequisites": prerequisites,
        "description": description,
        "department": department,
        "language": language,
        "degree": degree,
    }


_EECS_COURSE_EXPANSION = [
    _course("CSIE1003", "Web Programming", 3, "Spring", "Tuesday 10:00-12:00, Thursday 10:00-11:00", "Prof. Hsu Chia-Yu", ["CSIE1001"], "Client-side web development with HTML, CSS, JavaScript, accessibility, responsive layout, and API integration for student projects.", "Computer Science and Information Engineering"),
    _course("CSIE1004", "Digital Systems Laboratory", 1, "Fall", "Friday 09:00-12:00", "Prof. Lin Po-Han", ["CSIE1001"], "Hands-on digital logic laboratory covering gates, flip-flops, finite state machines, timing, and FPGA prototyping basics.", "Computer Science and Information Engineering"),
    _course("CSIE2003", "Database Systems", 3, "Fall", "Tuesday 13:00-16:00", "Prof. Yang Shu-Chen", ["CSIE2001"], "Relational models, SQL, indexing, transactions, normalization, query planning, and database-backed application design.", "Computer Science and Information Engineering"),
    _course("CSIE2004", "Software Engineering", 3, "Spring", "Wednesday 09:00-12:00", "Prof. Huang I-Min", ["CSIE2001"], "Requirements, architecture, design patterns, testing, version control, continuous integration, and team-based software delivery.", "Computer Science and Information Engineering"),
    _course("CSIE2005", "Object-Oriented Programming", 3, "Fall / Spring", "Monday 15:00-18:00", "Prof. Lai Ming-Yu", ["CSIE1001"], "Object-oriented design with classes, inheritance, interfaces, generics, exceptions, testing, and maintainable application structure.", "Computer Science and Information Engineering"),
    _course("CSIE2006", "Programming Languages", 3, "Spring", "Thursday 13:00-16:00", "Prof. Chen Li-Wei", ["CSIE2001"], "Syntax, semantics, type systems, functional programming, runtime environments, and comparative study of modern languages.", "Computer Science and Information Engineering"),
    _course("CSIE3003", "Computer Networks", 3, "Fall", "Tuesday 09:00-12:00", "Prof. Wu Kai-Sheng", ["CSIE2001", "CSIE2002"], "Layered network architecture, routing, TCP/IP, congestion control, socket programming, and network measurement experiments.", "Computer Science and Information Engineering"),
    _course("CSIE3004", "Information Security", 3, "Spring", "Wednesday 13:00-16:00", "Prof. Lee Fang-Ju", ["CSIE2001", "CSIE2002"], "Cryptography, authentication, access control, secure software, web security, malware concepts, and incident response.", "Computer Science and Information Engineering"),
    _course("CSIE3005", "Compiler Design", 3, "Fall", "Friday 10:00-13:00", "Prof. Chang Yu-Ting", ["CSIE2001", "CSIE2006"], "Lexical analysis, parsing, semantic analysis, intermediate representation, optimization, and code generation.", "Computer Science and Information Engineering"),
    _course("CSIE3006", "Human-Computer Interaction", 3, "Spring", "Monday 10:00-13:00", "Prof. Su Mei-Rong", ["CSIE1003"], "User research, interaction design, prototyping, usability evaluation, inclusive design, and interface analytics.", "Computer Science and Information Engineering", "English"),
    _course("CSIE3007", "Cloud Computing", 3, "Fall", "Thursday 09:00-12:00", "Prof. Tseng Hao-Chun", ["CSIE3003"], "Virtualization, containers, cloud storage, distributed deployment, observability, and resilient service operation.", "Computer Science and Information Engineering", "English"),
    _course("CSIE3008", "Embedded Systems", 3, "Spring", "Tuesday 14:00-17:00", "Prof. Kuo Cheng-Han", ["CSIE2002"], "Microcontrollers, real-time constraints, peripheral interfaces, embedded C, sensors, interrupts, and system debugging.", "Computer Science and Information Engineering"),
    _course("CSIE3009", "Parallel Programming", 3, "Fall", "Wednesday 14:00-17:00", "Prof. Chiu Wen-Liang", ["CSIE3001"], "Shared-memory and distributed parallel programming using threads, GPU kernels, synchronization, and performance profiling.", "Computer Science and Information Engineering", "English"),
    _course("CSIE3010", "Mobile Application Development", 3, "Spring", "Friday 14:00-17:00", "Prof. Lin Chih-Yuan", ["CSIE1003", "CSIE2005"], "Mobile UI patterns, local storage, networking, sensors, testing, deployment, and cross-platform application architecture.", "Computer Science and Information Engineering"),
    _course("CSIE3011", "Data Mining", 3, "Fall", "Monday 13:00-16:00", "Prof. Kao Hsin-Yi", ["CSIE2001", "MATH2002"], "Data preprocessing, association rules, classification, clustering, anomaly detection, recommender systems, and evaluation metrics.", "Computer Science and Information Engineering", "English"),
    _course("CSIE3012", "Web Backend Systems", 3, "Spring", "Thursday 15:00-18:00", "Prof. Tang Rui-Lin", ["CSIE1003", "CSIE2003"], "REST APIs, authentication, caching, background jobs, database integration, deployment, and server-side security.", "Computer Science and Information Engineering"),
    _course("CSIE4005", "Reinforcement Learning", 3, "Spring", "Tuesday 09:00-12:00", "Prof. Tsai Mei-Ling", ["CSIE4001"], "Markov decision processes, dynamic programming, temporal difference learning, policy gradients, and deep reinforcement learning.", "Computer Science and Information Engineering", "English"),
    _course("CSIE4006", "Big Data Analytics", 3, "Fall", "Wednesday 09:00-12:00", "Prof. Ho Yi-Chen", ["CSIE3011"], "Distributed data processing, Spark, stream analytics, feature pipelines, data lakes, and scalable analytics workflows.", "Computer Science and Information Engineering", "English"),
    _course("CSIE4007", "Distributed Systems", 3, "Spring", "Monday 09:00-12:00", "Prof. Liao Chien-Ming", ["CSIE3002", "CSIE3003"], "Replication, consensus, fault tolerance, distributed storage, service discovery, and large-scale system design.", "Computer Science and Information Engineering", "English"),
    _course("CSIE4008", "Blockchain and FinTech Security", 3, "Fall", "Friday 09:00-12:00", "Prof. Yeh Kuan-Ting", ["CSIE3004"], "Distributed ledgers, smart contracts, cryptographic protocols, wallet security, DeFi risks, and secure contract testing.", "Computer Science and Information Engineering"),
    _course("CSIE4009", "Robotics Software", 3, "Spring", "Wednesday 15:00-18:00", "Prof. Pan Li-Hua", ["CSIE3008", "CSIE4001"], "Robot operating systems, perception pipelines, localization, planning, control integration, and simulation-based testing.", "Computer Science and Information Engineering", "English"),
    _course("CSIE4010", "Edge AI Systems", 3, "Fall", "Thursday 13:00-16:00", "Prof. Fang Chih-Hao", ["CSIE3008", "CSIE4001"], "Model compression, on-device inference, sensor pipelines, privacy-aware deployment, and edge performance optimization.", "Computer Science and Information Engineering", "English"),
    _course("CSIE5001", "Advanced Algorithms", 3, "Fall", "Monday 14:00-17:00", "Prof. Chang Shu-Fen", ["CSIE3001"], "Graduate-level algorithm design and analysis covering approximation, randomized algorithms, online algorithms, and lower bounds.", "Computer Science and Information Engineering", "English", "master"),
    _course("CSIE5002", "Advanced Operating Systems", 3, "Spring", "Tuesday 14:00-17:00", "Prof. Liu Zhi-Yuan", ["CSIE3002"], "Kernel design, virtualization, storage systems, multicore scheduling, isolation, and research papers in operating systems.", "Computer Science and Information Engineering", "English", "master"),
    _course("CSIE5003", "Graph Machine Learning", 3, "Fall", "Wednesday 10:00-13:00", "Prof. Ko Wen-Jie", ["CSIE4001"], "Graph representation learning, graph neural networks, knowledge graphs, recommender graphs, and scalable graph training.", "Computer Science and Information Engineering", "English", "master"),
    _course("CSIE5004", "Secure Systems Engineering", 3, "Spring", "Thursday 10:00-13:00", "Prof. Lee Fang-Ju", ["CSIE3004"], "Threat modeling, secure architecture, program analysis, sandboxing, secure deployment, and systems security research.", "Computer Science and Information Engineering", "English", "master"),
    _course("CSIE5005", "Graduate Natural Language Processing", 3, "Fall", "Friday 13:00-16:00", "Prof. Ko Wen-Jie", ["CSIE4003"], "Advanced language modeling, retrieval-augmented generation, multilingual NLP, evaluation, alignment, and responsible deployment.", "Computer Science and Information Engineering", "English", "master"),
    _course("CSIE5006", "Cloud Native Applications", 3, "Spring", "Monday 18:00-21:00", "Prof. Tseng Hao-Chun", ["CSIE3007"], "Kubernetes, service meshes, observability, continuous delivery, API gateways, and production cloud application patterns.", "Computer Science and Information Engineering", "English", "master"),
    _course("CSIE7001", "Doctoral Research Seminar in Computing", 1, "Fall / Spring", "Friday 15:00-17:00", "Prof. Liu Pei-Shan", ["CSIE6001"], "PhD seminar for research proposal development, scholarly critique, conference presentation, and dissertation progress review.", "Computer Science and Information Engineering", "English", "phd"),
    _course("EE2001", "Circuit Theory", 3, "Fall", "Monday 09:00-12:00", "Prof. Cheng Yu-Lin", [], "Resistive circuits, nodal and mesh analysis, transient response, sinusoidal steady state, and frequency-domain circuit methods.", "Electrical Engineering"),
    _course("EE2002", "Signals and Systems", 3, "Spring", "Tuesday 10:00-13:00", "Prof. Lin Wan-Chen", ["MATH2001"], "Continuous and discrete signals, convolution, Fourier analysis, Laplace transforms, z-transforms, and system response.", "Electrical Engineering"),
    _course("EE2003", "Electronics I", 3, "Fall", "Wednesday 10:00-13:00", "Prof. Lai Sheng-Hao", ["EE2001"], "Diodes, MOSFETs, BJTs, biasing, small-signal models, amplifiers, and laboratory-oriented circuit analysis.", "Electrical Engineering"),
    _course("EE2004", "Electromagnetics", 3, "Spring", "Thursday 09:00-12:00", "Prof. Hwang Pei-Jung", ["MATH2001"], "Vector fields, Maxwell equations, transmission lines, wave propagation, and electromagnetic applications in EECS.", "Electrical Engineering"),
    _course("EE2005", "Logic Design", 3, "Fall / Spring", "Friday 09:00-12:00", "Prof. Chou Tzu-Han", ["CSIE1001"], "Boolean algebra, combinational logic, sequential circuits, timing, memory elements, and HDL-based design.", "Electrical Engineering"),
    _course("EE3001", "Microprocessors", 3, "Fall", "Monday 13:00-16:00", "Prof. Tsai Rong-Jie", ["EE2005"], "Microprocessor architecture, assembly programming, bus protocols, interrupts, timers, and embedded interface design.", "Electrical Engineering"),
    _course("EE3002", "Control Systems", 3, "Spring", "Tuesday 13:00-16:00", "Prof. Yeh Ming-De", ["EE2002"], "Feedback control, transfer functions, stability, root locus, frequency response, PID control, and state-space models.", "Electrical Engineering"),
    _course("EE3003", "VLSI Design", 3, "Fall", "Wednesday 14:00-17:00", "Prof. Shen Wei-Ting", ["EE2003", "EE2005"], "CMOS logic, layout design, timing, power, verification, and VLSI design flow for digital integrated circuits.", "Electrical Engineering", "English"),
    _course("EE3004", "Communication Systems", 3, "Spring", "Thursday 14:00-17:00", "Prof. Liu An-Chi", ["EE2002", "MATH2002"], "Analog and digital modulation, noise, channel models, receivers, coding concepts, and communication link analysis.", "Electrical Engineering"),
    _course("EE3005", "Power Electronics", 3, "Fall", "Friday 13:00-16:00", "Prof. Wang Chia-Hung", ["EE2001"], "Switching converters, rectifiers, inverters, magnetic components, control methods, and renewable power applications.", "Electrical Engineering"),
    _course("EE3006", "Semiconductor Devices", 3, "Spring", "Monday 10:00-13:00", "Prof. Wu Yen-Chi", ["EE2003"], "Carrier transport, pn junctions, MOS capacitors, transistors, device scaling, and semiconductor fabrication concepts.", "Electrical Engineering", "English"),
    _course("EE3007", "Digital Signal Processing", 3, "Fall", "Tuesday 15:00-18:00", "Prof. Lin Wan-Chen", ["EE2002"], "Discrete-time systems, DFT, FFT, filter design, spectral analysis, and real-time signal processing applications.", "Electrical Engineering", "English"),
    _course("EE3008", "FPGA System Design", 3, "Spring", "Wednesday 09:00-12:00", "Prof. Shen Wei-Ting", ["EE2005"], "HDL design, FPGA toolchains, timing closure, hardware debugging, memory interfaces, and accelerator prototyping.", "Electrical Engineering", "English"),
    _course("EE4001", "Integrated Circuit Design Laboratory", 2, "Fall", "Thursday 13:00-17:00", "Prof. Shen Wei-Ting", ["EE3003"], "Hands-on IC implementation lab covering layout, extraction, simulation, verification, and design review workflows.", "Electrical Engineering", "English"),
    _course("EE4002", "RF Circuit Design", 3, "Spring", "Friday 10:00-13:00", "Prof. Huang Li-Wei", ["EE2004", "EE2003"], "RF amplifiers, mixers, oscillators, impedance matching, noise, linearity, and wireless front-end design.", "Electrical Engineering", "English"),
    _course("EE4003", "Internet of Things Hardware", 3, "Fall", "Tuesday 09:00-12:00", "Prof. Tsai Rong-Jie", ["EE3001", "CSIE3003"], "IoT sensors, wireless modules, low-power firmware, hardware integration, edge gateways, and system prototyping.", "Electrical Engineering", "English"),
    _course("EE4004", "Renewable Energy Systems", 3, "Spring", "Wednesday 13:00-16:00", "Prof. Wang Chia-Hung", ["EE3005"], "Solar, wind, storage, grid integration, converters, energy management, and reliability of renewable energy systems.", "Electrical Engineering"),
    _course("EE5001", "Advanced Semiconductor Engineering", 3, "Fall", "Monday 09:00-12:00", "Prof. Wu Yen-Chi", ["EE3006"], "Advanced device physics, process integration, nanoscale effects, reliability, and semiconductor research methods.", "Electrical Engineering", "English", "master"),
    _course("EE5002", "Intelligent Control", 3, "Spring", "Tuesday 18:00-21:00", "Prof. Yeh Ming-De", ["EE3002"], "Adaptive control, optimal control, neural control, fuzzy systems, reinforcement learning control, and robotic applications.", "Electrical Engineering", "English", "master"),
    _course("EE5003", "Mixed-Signal IC Design", 3, "Fall", "Thursday 18:00-21:00", "Prof. Huang Li-Wei", ["EE3003"], "Data converters, PLLs, clocking, analog layout, noise analysis, and mixed-signal verification workflows.", "Electrical Engineering", "English", "master"),
    _course("EE6001", "EECS Research Methods", 2, "Fall / Spring", "Friday 10:00-12:00", "Prof. Liu Pei-Shan", [], "Research planning, literature review, reproducible experiments, technical writing, ethics, and presentation for EECS graduate students.", "Electrical Engineering", "English", "master"),
    _course("COMM2001", "Communication Networks", 3, "Fall", "Monday 15:00-18:00", "Prof. Hsieh Yu-Fang", ["CSIE2001"], "Network architectures, switching, routing, wireless access, performance metrics, and communication network simulation.", "Communication Engineering"),
    _course("COMM2002", "Probability for Communications", 3, "Spring", "Tuesday 09:00-12:00", "Prof. Ho Ming-Sheng", ["MATH2002"], "Random variables, stochastic processes, estimation, noise models, and probability tools for communication systems.", "Communication Engineering"),
    _course("COMM3001", "Wireless Communications", 3, "Fall", "Wednesday 10:00-13:00", "Prof. Liu An-Chi", ["EE3004"], "Wireless channels, modulation, OFDM, MIMO, link budgets, cellular systems, and wireless performance evaluation.", "Communication Engineering", "English"),
    _course("COMM3002", "Network Security", 3, "Spring", "Thursday 10:00-13:00", "Prof. Hsieh Yu-Fang", ["COMM2001", "CSIE3004"], "Secure routing, VPNs, intrusion detection, wireless security, protocol attacks, and practical network defense.", "Communication Engineering", "English"),
    _course("COMM3003", "Multimedia Signal Processing", 3, "Fall", "Friday 09:00-12:00", "Prof. Chen Pei-Ling", ["EE3007"], "Image, audio, and video processing, compression standards, streaming systems, and multimedia quality assessment.", "Communication Engineering"),
    _course("COMM4001", "5G Mobile Networks", 3, "Spring", "Monday 13:00-16:00", "Prof. Ho Ming-Sheng", ["COMM3001"], "5G radio access, core networks, slicing, edge computing, mobility management, and mobile network optimization.", "Communication Engineering", "English"),
    _course("COMM5001", "Advanced Wireless Networks", 3, "Fall", "Wednesday 18:00-21:00", "Prof. Liu An-Chi", ["COMM3001"], "Graduate study of wireless network protocols, resource allocation, mobile edge systems, and current research papers.", "Communication Engineering", "English", "master"),
]

RAW_COURSES.extend(_EECS_COURSE_EXPANSION)

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
    preferred_language: str | None = None
    language_priority: str | None = None

    @staticmethod
    def _extract_preferred_language(constraints: list[str]) -> str | None:
        language, _ = UserProfile._extract_language_preference(constraints)
        return language

    @staticmethod
    def _extract_language_priority(constraints: list[str]) -> str | None:
        _, priority = UserProfile._extract_language_preference(constraints)
        return priority

    @staticmethod
    def _extract_language_preference(constraints: list[str]) -> tuple[str | None, str | None]:
        for keyword, normalized in [
            ("english", "English"),
            ("chinese", "Chinese"),
        ]:
            matching = [constraint.lower() for constraint in constraints if keyword in constraint.lower()]
            if matching:
                hard_words = (
                    "only", "must", "required", "require", "mandatory",
                    "taught in", "taught by", "english-taught", "chinese-taught",
                )
                soft_words = ("prefer", "preferred", "preference", "better", "if possible")
                if any(any(word in constraint for word in hard_words) for constraint in matching):
                    return normalized, "required"
                if any(any(word in constraint for word in soft_words) for constraint in matching):
                    return normalized, "preferred"
                return normalized, "preferred"
        return None, None

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

    @staticmethod
    def _constraint_category(constraint: str) -> str:
        text = constraint.lower()
        category_terms = {
            "language": (
                "english", "chinese", "language", "taught",
                "english-taught", "chinese-taught",
            ),
            "schedule": (
                "schedule", "time", "morning", "afternoon", "evening", "night",
                "monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday", "weekend",
            ),
            "workload": (
                "credit", "credits", "workload", "light", "heavy",
                "easy", "hard", "intensive",
            ),
            "degree": (
                "undergrad", "undergraduate", "master", "graduate", "phd",
                "doctoral",
            ),
        }
        for category, terms in category_terms.items():
            if any(term in text for term in terms):
                return category
        return "general"

    @classmethod
    def _merge_constraints(cls, existing: list[str], incoming: list[str]) -> list[str]:
        cleaned = [constraint.strip() for constraint in incoming if constraint and constraint.strip()]
        if not cleaned:
            return existing

        removals = [
            constraint for constraint in cleaned
            if any(kw in constraint.lower() for kw in ("no longer", "not anymore", "removed", "remove"))
        ]
        if any(
            phrase in removal.lower()
            for removal in removals
            for phrase in ("all constraints", "any constraints", "no constraints")
        ):
            existing = []

        additions = [constraint for constraint in cleaned if constraint not in removals]
        replaced_categories = (
            {cls._constraint_category(constraint) for constraint in removals}
            | {cls._constraint_category(constraint) for constraint in additions}
        )

        merged = [
            constraint for constraint in existing
            if cls._constraint_category(constraint) not in replaced_categories
        ]
        for constraint in additions:
            category = cls._constraint_category(constraint)
            merged = [
                existing_constraint for existing_constraint in merged
                if cls._constraint_category(existing_constraint) != category
            ]
            if not any(constraint.lower() == existing_constraint.lower() for existing_constraint in merged):
                merged.append(constraint)
        return merged[-4:]

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
            self.constraints = self._merge_constraints(self.constraints, args["constraints"] or [])
            self.preferred_language = self._extract_preferred_language(self.constraints)
            self.language_priority = self._extract_language_priority(self.constraints)

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
        language     = self.preferred_language or "not specified"
        language_priority = self.language_priority or "none"

        return (
            f"Degree     : {degree_label}\n"
            f"Year       : {self.academic_year}\n"
            f"Language   : {language} ({language_priority})\n"
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
