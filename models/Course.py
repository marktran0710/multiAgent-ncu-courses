from dataclasses import dataclass

@dataclass
class Course:
    id: str
    name: str
    credits: int
    semester: str
    schedule: str
    instructor: str
    prerequisites: list[str]
    description: str
    department: str
    language: str = "Chinese"
    degree: str = "undergrad"

    def full_text(self) -> str:
        prereq_str = ", ".join(self.prerequisites) if self.prerequisites else "none"
        return (
            f"{self.id} {self.name} {self.department} "
            f"prerequisites {prereq_str} "
            f"language {self.language} degree {self.degree} "
            f"{self.description}"
        )

    def summary(self) -> str:
        prereq_str = ", ".join(self.prerequisites) if self.prerequisites else "None"
        return (
            f"[{self.id}] {self.name}\n"
            f"  Credits: {self.credits} | Semester: {self.semester}\n"
            f"  Schedule: {self.schedule}\n"
            f"  Language: {self.language} | Degree: {self.degree}\n"
            f"  Instructor: {self.instructor}\n"
            f"  Prerequisites: {prereq_str}\n"
            f"  Description: {self.description}\n"
        )