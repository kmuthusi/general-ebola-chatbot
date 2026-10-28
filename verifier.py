"""Validate that answers cite source pages present in retrieved evidence."""

import re

from bounded_composer import ERROR_PREFIX, REFUSAL


CITATION_RE = re.compile(r"\(([a-z0-9][a-z0-9._-]*\.pdf),\s*PDF Page\s+(\d+)\)", re.I)


def verify_answer(answer: str, evidence: list[dict]) -> dict:
    """Check citation format and source/page membership in retrieved evidence."""
    if answer == REFUSAL:
        return {"verified_answer": answer, "issues": []}
    if answer.startswith(ERROR_PREFIX):
        return {"verified_answer": REFUSAL, "issues": [answer]}

    allowed = {
        (str(item.get("source", "")).lower(), int(item["page"]))
        for item in evidence
        if item.get("source") and item.get("page") is not None
    }
    cited = {
        (source.lower(), int(page))
        for source, page in CITATION_RE.findall(answer)
    }
    issues = []
    if not cited:
        issues.append("The answer has no document and PDF page citation.")
    invalid = cited - allowed
    if invalid:
        issues.append(f"Citations outside retrieved evidence: {sorted(invalid)}")

    return {
        "verified_answer": answer if not issues else REFUSAL,
        "issues": issues,
    }
