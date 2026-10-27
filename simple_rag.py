"""CLI and shared orchestration for the WHO guidelines assistant."""

import argparse
import os
import sys
import time
from dataclasses import dataclass

from bounded_composer import (
    REFUSAL,
    compose_answer,
    normalize_generated_text,
    repair_citations,
    verify_claim_support,
)
from conversation import clarification_for, contextualize_question
from public_safety import CARE_REDIRECT, requires_professional_care
from rag_pipeline import exclude_bibliography, rerank_and_release
from retrievers import DEFAULT_SOURCE, GuidelineRetriever
from vector_store import VectorStore
from verifier import verify_answer


DEFAULT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


@dataclass
class SimpleAssistantRequest:
    question: str
    top_k: int = 5
    ollama_model: str = DEFAULT_MODEL
    earlier_questions: tuple[str, ...] = ()


class SimpleAssistantRAG:
    def __init__(self, source_path: str = DEFAULT_SOURCE, ollama_model: str = DEFAULT_MODEL):
        self.retriever = GuidelineRetriever(source_path)
        self.ollama_model = ollama_model
        self.store = VectorStore(collection_name="who_ebola_guidelines_ollama")

    def index(self) -> None:
        self.store.ensure_index(self.retriever.read_and_chunk())

    def ask(self, request: SimpleAssistantRequest) -> dict:
        if not request.question.strip():
            raise ValueError("Question must not be empty.")
        started = time.perf_counter()
        if requires_professional_care(request.question):
            return {"answer": CARE_REDIRECT, "issues": [], "chunks": [],
                    "elapsed_seconds": round(time.perf_counter() - started, 2)}
        clarification = clarification_for(request.question, request.earlier_questions)
        if clarification:
            return {"answer": clarification, "issues": [], "chunks": [],
                    "elapsed_seconds": round(time.perf_counter() - started, 2)}
        search_question = contextualize_question(request.question, request.earlier_questions)
        pages = self.retriever.read_and_chunk()
        self.store.ensure_index(pages)
        candidates = self.store.query(
            search_question, k=max(20, request.top_k * 4), include_scores=True
        )
        candidates = exclude_bibliography(search_question, candidates, pages)
        selected = rerank_and_release(
            search_question, candidates, intent="guideline", limit=request.top_k
        )
        evidence = selected["evidence"]
        if selected["release"] == "refusal":
            return {"answer": REFUSAL, "issues": [], "chunks": evidence,
                    "elapsed_seconds": round(time.perf_counter() - started, 2)}
        answer = normalize_generated_text(
            compose_answer(search_question, evidence, request.ollama_model)
        )
        checked = verify_answer(answer, evidence)
        if checked["issues"]:
            answer = normalize_generated_text(
                repair_citations(search_question, answer, evidence, request.ollama_model)
            )
            checked = verify_answer(answer, evidence)
        issues = list(checked["issues"])
        if not issues and checked["verified_answer"] != REFUSAL:
            supported, reason = verify_claim_support(
                checked["verified_answer"], evidence, request.ollama_model
            )
            if not supported:
                issues.append(reason)
        final_answer = REFUSAL if issues else checked["verified_answer"]
        return {"answer": final_answer, "issues": issues,
                "chunks": evidence if not issues else [], "elapsed_seconds": round(time.perf_counter() - started, 2)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="WHO Ebola Guidelines Assistant")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Directory of WHO PDFs or one PDF")
    parser.add_argument("--index", action="store_true", help="Build or refresh the local index")
    parser.add_argument("--ask", help="Ask a question about the WHO documents")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Installed Ollama chat model")
    parser.add_argument("--topk", type=int, default=5, help="Number of evidence passages")
    args = parser.parse_args()
    rag = SimpleAssistantRAG(args.source, args.model)
    if args.index:
        rag.index()
        print("Index ready.")
    if args.ask:
        result = rag.ask(SimpleAssistantRequest(args.ask, args.topk, args.model))
        print(f"\n{result['answer']}\n")
        if result["issues"]:
            print("Citation checks:", "; ".join(result["issues"]))
        for item in result["chunks"]:
            print(f"{item['source']}, PDF Page {item['page']}: {item['text'][:220]}...")


if __name__ == "__main__":
    main()
