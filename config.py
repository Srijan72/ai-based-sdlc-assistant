"""
LLM configuration for the SDLC Assistant.

Using Groq because it's free, extremely fast (LPU inference), and has a
first-class LangChain integration -- good fit for a student/demo project.
Get a free API key at: https://console.groq.com/keys

If you'd rather use OpenAI instead, see the commented block below.
"""

import os
from langchain_groq import ChatGroq

# Prefer environment variable; Streamlit UI can also set this at runtime.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Llama 3.3 70B versatile -- strong reasoning, generous free tier on Groq.
MODEL_NAME = "llama-3.3-70b-versatile"


def get_llm(api_key: str = None, temperature: float = 0.3):
    """Return a configured LangChain chat model.

    Each agent can request a different temperature (e.g. lower for the
    Reviewer/Tester, higher for creative requirement drafting).
    """
    key = api_key or GROQ_API_KEY
    if not key:
        raise ValueError(
            "No Groq API key found. Set GROQ_API_KEY env var, or enter it "
            "in the Streamlit sidebar."
        )
    return ChatGroq(
        api_key=key,
        model=MODEL_NAME,
        temperature=temperature,
    )


# --- To swap to OpenAI instead ---
# from langchain_openai import ChatOpenAI
# def get_llm(api_key=None, temperature=0.3):
#     return ChatOpenAI(
#         api_key=api_key or os.environ.get("OPENAI_API_KEY"),
#         model="gpt-4o-mini",
#         temperature=temperature,
#     )
