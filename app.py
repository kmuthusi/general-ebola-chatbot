"""Streamlit interface for the local Kenya Ebola guideline assistant."""

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread

import streamlit as st

from model_warmup import warm_models
from retrievers import DEFAULT_PDF, GuidelineRetriever
from simple_rag import DEFAULT_MODEL, SimpleAssistantRAG, SimpleAssistantRequest
from vector_store import DEFAULT_EMBED_MODEL


@st.cache_resource(show_spinner=False, ttl="25m")
def start_model_warmup(chat_model: str, embed_model: str, pdf_path: str) -> Future:
    """Prepare models and PDF without delaying the app page."""
    future = Future()

    def load() -> None:
        try:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:
                models = pool.submit(warm_models, chat_model, embed_model)
                pages = pool.submit(GuidelineRetriever(pdf_path).read_and_chunk)
                models.result()
                pages.result()
            future.set_result(round(time.perf_counter() - started, 2))
        except Exception as exc:
            future.set_exception(exc)

    Thread(target=load, daemon=True, name="ollama-model-warmup").start()
    return future


st.set_page_config(page_title="Kenya Ebola Case Management Guidelines Assistant")
st.title("Kenya Ebola Case Management Guidelines Assistant")
st.caption("Answers grounded in the supplied 2026 Kenya filovirus case management guideline.")
st.info(
    "Reference tool for trained health workers. For a suspected case or urgent clinical "
    "decision, follow your facility's current protocols and contact the appropriate clinical "
    "and public health response teams."
)

pdf_path = st.text_input("Guideline PDF path", DEFAULT_PDF)
model = st.text_input("Ollama chat model", DEFAULT_MODEL)
top_k = st.slider("Evidence passages", 1, 10, 5)
embed_model = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
warmup = start_model_warmup(model.strip(), embed_model, pdf_path)


@st.fragment(run_every="5s")
def show_warmup_status() -> None:
    if not warmup.done():
        st.caption("Loading local models and guideline in the background...")
        return
    try:
        seconds = warmup.result()
        st.caption(f"Models and guideline ready (loaded in {seconds:.1f} seconds).")
    except Exception as exc:
        st.warning(f"Model warmup failed: {exc}")
        if st.button("Retry model warmup"):
            start_model_warmup.clear()
            st.rerun()


show_warmup_status()

with st.form("question_form"):
    question = st.text_input("Ask about the guideline")
    submitted = st.form_submit_button("Ask")

if st.button("Build or refresh index"):
    try:
        with st.spinner("Reading and indexing the guideline..."):
            SimpleAssistantRAG(pdf_path, model).index()
        st.success("Index ready.")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

if submitted and question.strip():
    try:
        with st.spinner("Waiting for local models to load..."):
            warmup.result(timeout=120)
        with st.spinner("Finding guideline passages and preparing an answer..."):
            result = SimpleAssistantRAG(pdf_path, model).ask(
                SimpleAssistantRequest(question, top_k, model)
            )
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption(f"Response time: {result['elapsed_seconds']:.1f} seconds")
        if result["issues"]:
            st.warning("Citation check: " + "; ".join(result["issues"]))
        st.subheader("Evidence")
        for item in result["chunks"]:
            with st.expander(f"PDF Page {item['page']}"):
                st.write(item["text"])
    except Exception as exc:
        st.error(f"Unable to answer: {exc}")
