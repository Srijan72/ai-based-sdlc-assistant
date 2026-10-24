"""
Orchestrator / Supervisor for the Multi-Agent SDLC Assistant.

This is the piece that makes it "multi-agent" rather than just one long
prompt: each agent is called independently, its output becomes part of a
shared `state`, and the Code Agent <-> Reviewer Agent pair runs in a real
feedback loop (the reviewer can send the code back for revision) instead of
a single fixed pass. Swap this class's internals for LangGraph later if you
want a visual graph -- the agent functions in agents.py don't need to change.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from agents import (
    requirements_agent,
    design_agent,
    code_agent,
    reviewer_agent,
    test_agent,
    documentation_agent,
)


@dataclass
class SDLCState:
    idea: str = ""
    requirements: str = ""
    design: str = ""
    code: str = ""
    review_history: list = field(default_factory=list)
    tests: str = ""
    docs: str = ""


class SDLCOrchestrator:
    def __init__(self, api_key: str, max_review_cycles: int = 2):
        self.api_key = api_key
        self.max_review_cycles = max_review_cycles
        self.state = SDLCState()

    def run(self, idea: str, progress_callback: Optional[Callable[[str, str], None]] = None):
        """Run the full pipeline. progress_callback(stage_name, output) is
        called after each agent finishes, so a UI can render live progress."""

        def emit(stage, output):
            if progress_callback:
                progress_callback(stage, output)

        self.state = SDLCState(idea=idea)

        # 1. Requirements
        self.state.requirements = requirements_agent(idea, api_key=self.api_key)
        emit("requirements", self.state.requirements)

        # 2. Design
        self.state.design = design_agent(self.state.requirements, api_key=self.api_key)
        emit("design", self.state.design)

        # 3 & 4. Code <-> Review loop
        feedback = None
        for cycle in range(1, self.max_review_cycles + 1):
            code = code_agent(
                self.state.requirements,
                self.state.design,
                api_key=self.api_key,
                revision_feedback=feedback,
            )
            self.state.code = code
            emit(f"code (attempt {cycle})", code)

            review = reviewer_agent(self.state.requirements, code, api_key=self.api_key)
            self.state.review_history.append(review)
            emit(f"review (attempt {cycle})", review["raw"])

            if review["verdict"] == "APPROVED":
                break
            feedback = review["feedback"]
        # After the loop, self.state.code holds the last generated version
        # (approved, or best effort after max_review_cycles).

        # 5. Tests
        self.state.tests = test_agent(self.state.requirements, self.state.code, api_key=self.api_key)
        emit("tests", self.state.tests)

        # 6. Documentation
        self.state.docs = documentation_agent(
            idea, self.state.requirements, self.state.design, self.state.code, api_key=self.api_key
        )
        emit("docs", self.state.docs)

        return self.state
