# AI-Based SDLC Assistant with Multi-Agent Architecture

A Streamlit app where 6 LangChain-powered agents collaborate to take a raw
software idea through the SDLC: **Requirements → Design → Code → Review →
Tests → Documentation**. The Code Agent and Reviewer Agent run in a real
feedback loop — if the reviewer rejects the code, it goes back for revision
(up to N cycles, configurable in the sidebar).

## Architecture

```
User Idea
   │
   ▼
[Requirements Analyst Agent] ──▶ requirements.md
   │
   ▼
[System Design Agent] ──▶ design.md
   │
   ▼
┌─────────────────────────────┐
│  [Code Generation Agent] ──▶ code            │
│         │                                    │
│         ▼                                    │
│  [Reviewer Agent] ──▶ APPROVED / REVISE ─────┤ (loops back with feedback)
└─────────────────────────────┘
   │ (approved / max cycles reached)
   ▼
[Test Generation Agent] ──▶ pytest suite
   │
   ▼
[Documentation Agent] ──▶ README.md
```

Each agent lives in `agents.py` as an independent LangChain LCEL chain
(`prompt | llm | parser`). The `orchestrator.py` module coordinates them and
owns the shared state — agents never call each other directly, which keeps
them easy to test, replace, or reorder.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a **free** Groq API key: https://console.groq.com/keys
   (Groq is used because it's free and fast; see `config.py` to swap in
   OpenAI or another provider instead.)

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Paste your API key into the sidebar, type a project idea, and click
   **Run SDLC Pipeline**.

## Project structure

```
sdlc_assistant/
├── app.py            # Streamlit UI
├── orchestrator.py   # Coordinates agents + code/review feedback loop
├── agents.py          # 6 agent definitions (LCEL chains)
├── config.py          # LLM setup (Groq by default)
├── requirements.txt
└── README.md
```

## Extending this project

- **Swap the orchestration engine**: replace `orchestrator.py`'s manual loop
  with **LangGraph** for a visual state graph — the agent functions in
  `agents.py` don't need to change.
- **Add more agents**: e.g. a Deployment/DevOps Agent that generates a
  Dockerfile, or a Security Review Agent.
- **Persist runs**: save each `SDLCState` to disk/DB so past projects can be
  revisited.
- **Multi-file code generation**: currently the Code Agent produces a single
  module; you could extend it to generate a whole file tree.
