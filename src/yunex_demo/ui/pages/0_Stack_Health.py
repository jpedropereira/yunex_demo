"""Stack Health -- proves the infrastructure works end to end.

Deliberately thin: every check lives in ``yunex_demo.health`` so the same logic
backs ``python -m yunex_demo.health`` on the command line.
"""

import streamlit as st

from yunex_demo import health
from yunex_demo.config import settings

st.set_page_config(
    page_title="Stack Health", page_icon=":material/health_and_safety:", layout="wide"
)

st.title("Stack Health")
st.caption("Run this before presenting. Every row must pass.")

if st.button("Run checks", type="primary", icon=":material/play_arrow:"):
    st.session_state.pop("health_results", None)

if "health_results" not in st.session_state:
    results = []
    progress = st.progress(0.0, text="Starting checks...")
    for i, check in enumerate(health.CHECKS):
        progress.progress(i / len(health.CHECKS), text=f"Running: {check.__name__}...")
        results.append(check())
    progress.empty()
    st.session_state["health_results"] = results

results = st.session_state["health_results"]
failed = [r for r in results if not r.ok]

if failed:
    st.error(f"{len(failed)} of {len(results)} checks failed.")
else:
    st.success(f"All {len(results)} checks passed -- the stack is ready.")

st.divider()

for r in results:
    icon = ":material/check_circle:" if r.ok else ":material/error:"
    colour = "green" if r.ok else "red"
    with st.container(border=True):
        head, timing = st.columns([6, 1])
        with head:
            st.markdown(f"{icon} **:{colour}[{r.name}]**")
            st.write(r.summary)
        with timing:
            st.caption(f"{r.elapsed_s:.2f}s")
        if r.error:
            st.code(r.error, language="text")
        if r.details:
            with st.expander("Details"):
                st.json(r.details)

st.divider()

with st.expander("Resolved configuration"):
    st.json(settings.model_dump())

st.caption(
    "A `gpu` profile on a host without nvidia-container-toolkit degrades silently "
    "to CPU -- the Acceleration row reports what Ollama actually loaded, so that "
    "case is visible here rather than as unexplained slowness on stage."
)
