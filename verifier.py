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


def filter_verified_format(answer: str, evidence: list[dict]) -> dict:
    """Keep well-formed bullets whose citations identify retrieved pages."""
    if answer == REFUSAL:
        return {"verified_answer": REFUSAL, "issues": []}
    allowed = {(item["source"].lower(), item["page"]) for item in evidence}
    issues = []
    verified_lines = []
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    for number, line in enumerate(lines, start=1):
        citations = CITATION_RE.findall(line)
        if not line.startswith("- "):
            issues.append(f"Claim {number} was not formatted as a bullet.")
            continue
        if len(citations) != 1 or not END_CITATION_RE.search(line):
            issues.append(f"Claim {number} did not have one final page citation.")
            continue
        source, page = citations[0]
        if (source.lower(), int(page)) not in allowed:
            issues.append(f"Claim {number} cited a page outside retrieved evidence.")
            continue
        verified_lines.append(line)
    return {
        "verified_answer": "\n".join(verified_lines) if verified_lines else REFUSAL,
        "issues": issues,
    }
