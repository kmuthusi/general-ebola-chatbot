"""Streamlit interface for the Kenya Ebola Care Guide."""

import os
import time
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Thread

import streamlit as st

from model_warmup import warm_models
from public_safety import CARE_REDIRECT, requires_professional_care
from retrievers import DEFAULT_PDF, GuidelineRetriever
from simple_rag import DEFAULT_MODEL, SimpleAssistantRAG, SimpleAssistantRequest
from vector_store import DEFAULT_EMBED_MODEL


LOGGER = logging.getLogger(__name__)


@st.cache_resource
def answer_slots() -> BoundedSemaphore:
    return BoundedSemaphore(max(1, int(os.getenv("MAX_CONCURRENT_ANSWERS", "2"))))


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
            LOGGER.exception("Guideline assistant startup failed")
            future.set_exception(exc)

    Thread(target=load, daemon=True, name="ollama-model-warmup").start()
    return future


st.set_page_config(page_title="Kenya Ebola Care Guide")
st.title("Kenya Ebola Care Guide")
st.caption("Ask about Kenya's 2026 filovirus case management guideline.")
st.info(
    "This assistant uses the Government of Kenya's October 2026 *Case Management Guidelines "
    "for Filovirus Disease*. It explains the document's guidance for health workers on Ebola "
    "and related filovirus diseases, with PDF page citations. It cannot diagnose illness or "
    "direct individual care. If infection is suspected, contact a health facility or public "
    "health response team promptly."
)
st.caption("Please do not enter names, contact details, or personal medical records.")

pdf_path = os.getenv("GUIDELINE_PDF_PATH", DEFAULT_PDF)
model = DEFAULT_MODEL
top_k = max(1, min(10, int(os.getenv("EVIDENCE_PASSAGES", "5"))))
embed_model = os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
warmup = start_model_warmup(model, embed_model, pdf_path)


@st.fragment(run_every="5s")
def show_warmup_status() -> None:
    if not warmup.done():
        st.caption("Preparing the guideline assistant...")
        return
    try:
        warmup.result()
        st.caption("Guideline assistant ready.")
    except Exception:
        st.warning("The guideline assistant is temporarily unavailable.")
        if st.button("Try again"):
            start_model_warmup.clear()
            st.rerun()


show_warmup_status()

with st.form("question_form"):
    question = st.text_input("Ask about the guideline", max_chars=500)
    submitted = st.form_submit_button("Ask")

if submitted and requires_professional_care(question):
    st.info(CARE_REDIRECT)
elif submitted and question.strip():
    slots = answer_slots()
    if not slots.acquire(blocking=False):
        st.warning("The assistant is busy. Please try again shortly.")
    else:
        try:
            with st.spinner("Checking the guideline..."):
                warmup.result(timeout=180)
                result = SimpleAssistantRAG(pdf_path, model).ask(
                    SimpleAssistantRequest(question, top_k, model)
                )
            st.subheader("Answer")
            st.write(result["answer"])
            if result["issues"]:
                st.warning("The answer could not be verified against the guideline. Review the passages below.")
            st.subheader("Source passages")
            for item in result["chunks"]:
                with st.expander(f"PDF Page {item['page']}"):
                    st.write(item["text"])
        except Exception:
            LOGGER.exception("Guideline question failed")
            st.error("The guideline assistant is temporarily unavailable. Please try again later.")
        finally:
            slots.release()
