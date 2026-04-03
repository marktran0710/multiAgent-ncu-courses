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

    def full_text(self) -> str:
        prereq_str = ", ".join(self.prerequisites) if self.prerequisites else "none"
        return (
            f"{self.id} {self.name} {self.department} "
            f"prerequisites {prereq_str} "
            f"{self.description}"
        )

    def summary(self) -> str:
        prereq_str = ", ".join(self.prerequisites) if self.prerequisites else "None"
        return (
            f"[{self.id}] {self.name}\n"
            f"  Credits: {self.credits} | Semester: {self.semester}\n"
            f"  Schedule: {self.schedule}\n"
            f"  Instructor: {self.instructor}\n"
            f"  Prerequisites: {prereq_str}\n"
            f"  Description: {self.description}\n"
        )