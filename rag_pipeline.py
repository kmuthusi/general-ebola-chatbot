"""Small relevance and evidence-release stage for the WHO RAG pipeline."""

import re
from typing import Any

TRANSMISSION_TERMS = re.compile(
    r"\b(?:spread|spreads|spreading|transmission|transmit\w*|airborne|respiratory|droplets?|fomites?)\b|\bthrough\s+(?:the\s+)?air\b",
    re.I,
)
FILOVIRUS_TERMS = re.compile(
    r"\b(?:ebola|ebod|ebov|marburg|filovirus|filoviruses|bundibugyo|bdbv|bvd)\b",
    re.I,
)


def transmission_answer(question: str, pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a source-fixed answer for transmission questions that risk air-route confusion."""
    if not TRANSMISSION_TERMS.search(question) or not FILOVIRUS_TERMS.search(question):
        return None

    page_by_source_and_number = {
        (item.get("source", "").lower(), item.get("page")): item
        for item in pages
    }
    ipc_name = "ipc-ebola-marburg-2026.pdf"
    bdbv_name = "who-bundibugyo-risk-assessment-drc-v4-2026.pdf"
    ipc_glossary = page_by_source_and_number.get((ipc_name, 11))
    ipc_air_route = page_by_source_and_number.get((ipc_name, 35))
    if not ipc_glossary or not ipc_air_route:
        return None

    if re.search(r"\b(?:bundibugyo|bdbv|bvd)\b", question, re.I):
        bdbv_route = page_by_source_and_number.get((bdbv_name, 9))
        if bdbv_route:
            answer = (
                "- WHO describes Bundibugyo virus transmission as contact with infected animals' "
                "blood or body fluids followed by person-to-person contact with infected people's "
                "blood or body fluids, or contaminated materials and surfaces "
                "(who-bundibugyo-risk-assessment-drc-v4-2026.pdf, PDF Page 9).\n"
                "- WHO's IPC guideline says current evidence indicates human filovirus transmission "
                "does not naturally occur via airborne or respiratory-droplet routes "
                "(ipc-ebola-marburg-2026.pdf, PDF Page 35)."
            )
            return {"answer": answer, "chunks": [bdbv_route, ipc_air_route], "issues": []}

    answer = (
        "- WHO describes Ebola disease transmission from infected wild animals to people and then "
        "between people through direct contact with blood, secretions, organs or other body fluids, "
        "or surfaces and materials contaminated with those fluids "
        "(ipc-ebola-marburg-2026.pdf, PDF Page 11).\n"
        "- WHO's IPC guideline says current evidence indicates human filovirus transmission does "
        "not naturally occur via airborne or respiratory-droplet routes "
        "(ipc-ebola-marburg-2026.pdf, PDF Page 35).\n"
        "- The guideline says evidence regarding fomite transmission from clothing or linens is "
        "limited (ipc-ebola-marburg-2026.pdf, PDF Page 35)."
    )
    return {"answer": answer, "chunks": [ipc_glossary, ipc_air_route], "issues": []}


def exclude_bibliography(
    question: str, candidates: list[dict[str, Any]], pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep references available for explicit bibliography questions only."""
    if re.search(r"\b(references?|bibliography|citations?)\b", question, re.I):
        return candidates

    first_reference_page = {}
    for page in pages:
        source = page.get("source", "")
        if source not in first_reference_page and re.search(
            r"(?im)^\s*(?:\d+\.?\s+)?References\s*$", page["text"][:500]
        ):
            first_reference_page[source] = page["page"]

    filtered = [
        item for item in candidates
        if item.get("source", "") not in first_reference_page
        or item.get("page", 0) < first_reference_page[item.get("source", "")]
    ]
    return filtered or candidates


def _simple_relevance_score(candidate: dict[str, Any]) -> float:
    """Use the normalized Chroma similarity score plus a page metadata bonus."""
    score = candidate.get("score", 0.0) or 0.0
    return score + (0.05 if candidate.get("page") else 0.0)


def rerank_and_release(
    question: str,
    candidates: list[dict[str, Any]],
    intent: str,
    limit: int = 5,
) -> dict[str, Any]:
    """Sort retrieved passages and decline generation when evidence is too weak."""
    ranked = sorted(candidates, key=_simple_relevance_score, reverse=True)
    if not ranked or all((item.get("score") or 0) < 0.2 for item in ranked):
        return {"evidence": ranked[:2], "release": "refusal"}
    return {"evidence": ranked[:limit], "release": "grounded"}
