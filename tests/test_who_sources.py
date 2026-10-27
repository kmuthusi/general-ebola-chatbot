from pathlib import Path

import pytest

import simple_rag
from bounded_composer import REFUSAL
from rag_pipeline import exclude_bibliography
from retrievers import GuidelineRetriever, _read_pdf
from simple_rag import SimpleAssistantRAG, SimpleAssistantRequest
from vector_store import _hash_manifest
from verifier import verify_answer


def test_all_who_pdfs_are_read_with_distinct_sources():
    directory = Path(__file__).resolve().parents[1] / "who_guidelines"
    _read_pdf.cache_clear()
    pages = GuidelineRetriever(str(directory)).read_and_chunk()
    assert {item["source"] for item in pages} == {file.name for file in directory.glob("*.pdf")}
    assert all(item["page"] >= 1 and item["text"] for item in pages)


def test_empty_directory_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No WHO guideline PDFs"):
        GuidelineRetriever(str(tmp_path)).read_and_chunk()


def test_document_identity_changes_index_manifest():
    a = [{"source": "a.pdf", "page": 1, "text": "Same text"}]
    b = [{"source": "b.pdf", "page": 1, "text": "Same text"}]
    assert _hash_manifest(a) != _hash_manifest(b)


def test_citation_must_identify_correct_document_and_page():
    evidence = [{"source": "a.pdf", "page": 3, "text": "Isolation guidance"}]
    assert not verify_answer("Isolate (a.pdf, PDF Page 3).", evidence)["issues"]
    assert verify_answer("Isolate (b.pdf, PDF Page 3).", evidence)["verified_answer"] == REFUSAL
    assert verify_answer("Isolate (a.pdf, PDF Page 4).", evidence)["verified_answer"] == REFUSAL
    assert verify_answer("Isolate (PDF Page 3).", evidence)["verified_answer"] == REFUSAL


def test_orchestrator_uses_document_aware_retrieval(monkeypatch):
    evidence = {"source": "ipc-ebola-marburg-2026.pdf", "page": 12,
                "text": "Relevant isolation text.", "score": 0.8}

    class FakeStore:
        def __init__(self, **kwargs):
            assert kwargs["collection_name"] == "who_ebola_guidelines_ollama"

        def ensure_index(self, pages):
            assert pages[0]["source"] == evidence["source"]

        def query(self, *args, **kwargs):
            return [evidence]

    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    monkeypatch.setattr(GuidelineRetriever, "read_and_chunk", lambda self: [evidence])
    monkeypatch.setattr(simple_rag, "compose_answer", lambda *args:
                        "See IPC guidance (ipc-ebola-marburg-2026.pdf, PDF Page 12).")
    result = SimpleAssistantRAG("who_guidelines").ask(SimpleAssistantRequest("What is IPC guidance?"))
    assert result["issues"] == []
    assert result["chunks"][0]["source"] == evidence["source"]


def test_bibliography_filter_uses_each_document_page_range():
    pages = [
        {"source": "a.pdf", "page": 3, "text": "Care instructions"},
        {"source": "a.pdf", "page": 9, "text": "References\nCitation"},
        {"source": "b.pdf", "page": 9, "text": "Care instructions"},
    ]
    candidates = [
        {"source": "a.pdf", "page": 9, "text": "Citation", "score": 0.9},
        {"source": "b.pdf", "page": 9, "text": "Care", "score": 0.8},
    ]
    assert exclude_bibliography("How is Ebola managed?", candidates, pages) == candidates[1:]
