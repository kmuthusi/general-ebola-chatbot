"""Route obvious individual care requests away from the public chatbot."""

import re


CARE_REDIRECT = (
    "This assistant can explain what the guideline covers, but it cannot guide "
    "an individual patient's care or provide treatment instructions. If Ebola "
    "is suspected, contact a health facility or local public health response "
    "team promptly."
)


def requires_professional_care(question: str) -> bool:
    text = re.sub(r"\s+", " ", question.lower()).strip()
    patterns = (
        r"\b(?:how|what)\s+(?:should|do|can)\s+(?:i|we)\b.{0,80}\b(?:manage|treat|care|give|administer|do)\b",
        r"\bhow\s+(?:to|do\s+you)\s+(?:manage|treat|care for|administer|give)\b",
        r"\b(?:dose|doses|dosing|dosage|prescribe|prescribing|how much|infusion rate|fluid rate|iv access)\b",
        r"\b(?:my|our)\s+(?:child|patient|relative|family member)\b",
        r"\bi\s+(?:have|am experiencing|think i have)\b.{0,80}\b(?:ebola|fever|bleeding|vomiting|diarrhoea|diarrhea)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)
