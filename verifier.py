"""Require citations that identify both a WHO document and a retrieved PDF page."""

import re

from bounded_composer import REFUSAL


CITATION_RE = re.compile(r"\(([a-z0-9][a-z0-9._-]*\.pdf),\s*PDF Page\s+(\d+)\)", re.I)
END_CITATION_RE = re.compile(r"\([a-z0-9][a-z0-9._-]*\.pdf,\s*PDF Page\s+\d+\)[.!?]*$", re.I)


def verify_answer(answer: str, evidence: list[dict]) -> dict:
    if answer == REFUSAL:
        return {"verified_answer": answer, "issues": []}
    cited = {(source.lower(), int(page)) for source, page in CITATION_RE.findall(answer)}
    allowed = {(item["source"].lower(), item["page"]) for item in evidence}
    issues = []
    if not cited:
        issues.append("The answer has no document and PDF page citation.")
    if cited - allowed:
        issues.append(f"Citations outside retrieved evidence: {sorted(cited - allowed)}")
    if any(
        line.lstrip().startswith(("- ", "* ", "• "))
        and not END_CITATION_RE.search(line.rstrip())
        for line in answer.splitlines()
    ):
        issues.append("Every answer bullet needs a document and PDF page citation.")
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines or any(not line.startswith("- ") for line in lines):
        issues.append("The answer must use one plain-text bullet per claim.")
    if any(not END_CITATION_RE.search(line) for line in lines):
        issues.append("Every claim must end with its document and PDF page citation.")
    return {"verified_answer": answer if not issues else REFUSAL, "issues": issues}
