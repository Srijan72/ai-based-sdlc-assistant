"""
Streamlit UI for the AI-Based SDLC Assistant (Multi-Agent Architecture).

Run with:
    streamlit run app.py
"""

import re
import streamlit as st
from orchestrator import SDLCOrchestrator

st.set_page_config(page_title="AI SDLC Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI-Based SDLC Assistant")
st.caption("Multi-Agent Architecture — Requirements → Design → Code → Review → Tests → Docs")

# --- Sidebar: config ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at https://console.groq.com/keys",
    )
    max_cycles = st.slider("Max code review cycles", 1, 4, 2)
    st.markdown("---")
    st.markdown(
        "**Pipeline**\n"
        "1. Requirements Analyst Agent\n"
        "2. System Design Agent\n"
        "3. Code Generation Agent\n"
        "4. Code Reviewer Agent (feedback loop)\n"
        "5. Test Generation Agent\n"
        "6. Documentation Agent"
    )

# --- Main input ---
idea = st.text_area(
    "Describe your software idea",
    placeholder="e.g. A web app where users can track daily expenses, "
    "categorize them, and view monthly spending charts.",
    height=120,
)

run_clicked = st.button("🚀 Run SDLC Pipeline", type="primary", disabled=not idea)

def extract_code_block(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


if run_clicked:
    if not api_key:
        st.error("Enter your Groq API key in the sidebar first.")
        st.stop()

    orchestrator = SDLCOrchestrator(api_key=api_key, max_review_cycles=max_cycles)

    stage_placeholders = {}
    stage_order = [
        "requirements", "design", "code", "review", "tests", "docs"
    ]
    tabs = st.tabs(
        ["📋 Requirements", "🏗️ Design", "💻 Code", "🔍 Review", "🧪 Tests", "📄 Docs"]
    )
    tab_map = dict(zip(stage_order, tabs))
    status = st.status("Running multi-agent pipeline...", expanded=True)

    def on_progress(stage: str, output: str):
        base = stage.split(" ")[0]  # "code (attempt 2)" -> "code"
        status.write(f"✅ {stage} done")
        target_tab = tab_map.get(base)
        if target_tab:
            with target_tab:
                if base == "code":
                    st.code(extract_code_block(output), language="python")
                elif base == "tests":
                    st.code(extract_code_block(output), language="python")
                elif base == "review":
                    st.text(output)
                else:
                    st.markdown(output)

    with st.spinner("Agents are working..."):
        final_state = orchestrator.run(idea, progress_callback=on_progress)

    status.update(label="Pipeline complete ✅", state="complete", expanded=False)

    st.markdown("---")
    st.subheader("⬇️ Downloads")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download code (.py)",
            extract_code_block(final_state.code),
            file_name="generated_module.py",
        )
    with col2:
        st.download_button(
            "Download tests (.py)",
            extract_code_block(final_state.tests),
            file_name="test_generated_module.py",
        )
    with col3:
        st.download_button(
            "Download README (.md)",
            final_state.docs,
            file_name="README.md",
        )
else:
    st.info("Enter an idea and click **Run SDLC Pipeline** to generate "
             "requirements, design, code, tests, and docs automatically.")
