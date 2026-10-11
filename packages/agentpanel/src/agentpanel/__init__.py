"""Domain-agnostic multi-agent orchestration core."""

from agentpanel.agents import Coordinator, Reviewer, Specialist
from agentpanel.graph import Workflow, WorkflowResult
from agentpanel.llm_agents import LLMReviewer, LLMSpecialist
from agentpanel.state import AgentContext, Artifact, Review, Verdict

__version__ = "0.1.0"

__all__ = [
    "AgentContext",
    "Artifact",
    "Coordinator",
    "LLMReviewer",
    "LLMSpecialist",
    "Review",
    "Reviewer",
    "Specialist",
    "Verdict",
    "Workflow",
    "WorkflowResult",
]
