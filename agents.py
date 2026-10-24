"""
Agent definitions for the Multi-Agent SDLC Assistant.

Each agent = (prompt template) + (LLM) + (output parser), wired with LCEL
(the `prompt | llm | parser` pipe syntax). Agents are independent and only
talk to each other through the shared `state` dict managed by the
Orchestrator (see orchestrator.py) -- this keeps each agent testable and
swappable on its own.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm


def _chain(system_prompt: str, human_template: str, api_key: str, temperature: float):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_template)]
    )
    llm = get_llm(api_key=api_key, temperature=temperature)
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 1. Requirements Analyst Agent
# ---------------------------------------------------------------------------
def requirements_agent(idea: str, api_key: str = None) -> str:
    chain = _chain(
        system_prompt=(
            "You are a senior Business/Requirements Analyst in a software "
            "development team. You convert a rough project idea into a "
            "clear, structured requirements document."
        ),
        human_template=(
            "Project idea:\n{idea}\n\n"
            "Produce a structured requirements document with these sections:\n"
            "1. Project Overview\n"
            "2. Functional Requirements (numbered list, FR-1, FR-2, ...)\n"
            "3. Non-Functional Requirements (performance, security, scalability)\n"
            "4. User Stories (as: 'As a <role>, I want <goal>, so that <reason>')\n"
            "5. Assumptions & Constraints\n"
            "Keep it concise and use Markdown."
        ),
        api_key=api_key,
        temperature=0.4,
    )
    return chain.invoke({"idea": idea})


# ---------------------------------------------------------------------------
# 2. System Design / Architect Agent
# ---------------------------------------------------------------------------
def design_agent(requirements: str, api_key: str = None) -> str:
    chain = _chain(
        system_prompt=(
            "You are a Software Architect. You turn requirements into a "
            "concrete, buildable technical design."
        ),
        human_template=(
            "Requirements document:\n{requirements}\n\n"
            "Produce a design document with these sections:\n"
            "1. High-Level Architecture (describe components/modules and how "
            "they interact; include a simple text/ASCII diagram)\n"
            "2. Recommended Tech Stack (with brief justification)\n"
            "3. Module Breakdown (list each module/class with its responsibility)\n"
            "4. Data Model (key entities and fields, if applicable)\n"
            "5. Key Design Decisions & Trade-offs\n"
            "Use Markdown."
        ),
        api_key=api_key,
        temperature=0.3,
    )
    return chain.invoke({"requirements": requirements})


# ---------------------------------------------------------------------------
# 3. Code Generation Agent
# ---------------------------------------------------------------------------
def code_agent(
    requirements: str,
    design: str,
    api_key: str = None,
    revision_feedback: str = None,
) -> str:
    human_template = (
        "Requirements:\n{requirements}\n\nDesign:\n{design}\n\n"
        "Generate working Python code that implements the core module(s) "
        "described in the design. Requirements:\n"
        "- Include necessary imports\n"
        "- Add docstrings and inline comments\n"
        "- Include a small `if __name__ == '__main__':` usage example\n"
        "- Return ONLY code, inside a single ```python fenced block, no prose."
    )
    if revision_feedback:
        human_template += (
            "\n\nIMPORTANT: A code reviewer rejected your previous version "
            "with this feedback. Fix these issues:\n{revision_feedback}"
        )

    chain = _chain(
        system_prompt=(
            "You are a senior Python Software Engineer. You write clean, "
            "correct, well-documented, production-quality code."
        ),
        human_template=human_template,
        api_key=api_key,
        temperature=0.2,
    )
    payload = {"requirements": requirements, "design": design}
    if revision_feedback:
        payload["revision_feedback"] = revision_feedback
    return chain.invoke(payload)


# ---------------------------------------------------------------------------
# 4. Code Reviewer Agent (the "second opinion" agent that closes the loop)
# ---------------------------------------------------------------------------
def reviewer_agent(requirements: str, code: str, api_key: str = None) -> dict:
    chain = _chain(
        system_prompt=(
            "You are a strict Code Reviewer / QA Lead. You check generated "
            "code against requirements for correctness, completeness, and "
            "quality."
        ),
        human_template=(
            "Requirements:\n{requirements}\n\nCode to review:\n{code}\n\n"
            "Respond in EXACTLY this format:\n"
            "VERDICT: APPROVED or REVISE\n"
            "FEEDBACK: <bullet list of issues, or 'None' if approved>"
        ),
        api_key=api_key,
        temperature=0.1,
    )
    result = chain.invoke({"requirements": requirements, "code": code})
    verdict = "APPROVED" if "VERDICT: APPROVED" in result.upper() else "REVISE"
    feedback = result.split("FEEDBACK:", 1)[-1].strip() if "FEEDBACK:" in result else result
    return {"verdict": verdict, "feedback": feedback, "raw": result}


# ---------------------------------------------------------------------------
# 5. Test Case Generation Agent
# ---------------------------------------------------------------------------
def test_agent(requirements: str, code: str, api_key: str = None) -> str:
    chain = _chain(
        system_prompt=(
            "You are a QA/Test Engineer specializing in Python unit testing "
            "with pytest."
        ),
        human_template=(
            "Requirements:\n{requirements}\n\nCode under test:\n{code}\n\n"
            "Generate a pytest test suite covering normal cases, edge cases, "
            "and invalid input. Return ONLY code, inside a single "
            "```python fenced block, no prose."
        ),
        api_key=api_key,
        temperature=0.2,
    )
    return chain.invoke({"requirements": requirements, "code": code})


# ---------------------------------------------------------------------------
# 6. Documentation Agent
# ---------------------------------------------------------------------------
def documentation_agent(
    idea: str, requirements: str, design: str, code: str, api_key: str = None
) -> str:
    chain = _chain(
        system_prompt=(
            "You are a Technical Writer who produces clear README-style "
            "documentation for developers."
        ),
        human_template=(
            "Project idea: {idea}\n\nRequirements:\n{requirements}\n\n"
            "Design:\n{design}\n\nCode:\n{code}\n\n"
            "Write a README.md with: Project Title, Description, Features, "
            "Tech Stack, Setup/Installation, Usage, and Architecture Summary. "
            "Use Markdown."
        ),
        api_key=api_key,
        temperature=0.3,
    )
    return chain.invoke(
        {"idea": idea, "requirements": requirements, "design": design, "code": code}
    )
