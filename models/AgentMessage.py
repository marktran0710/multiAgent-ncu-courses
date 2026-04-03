from dataclasses import dataclass


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    payload: dict[str, Any]
