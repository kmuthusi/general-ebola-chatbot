"""Route obvious individual care requests away from the public chatbot."""

import re


CARE_REDIRECT = (
    "This assistant can explain what the WHO guidelines cover, but it cannot guide "
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
        r"\b(?:guidance|recommendations?|protocol|instructions?)\b.{0,100}\b(?:assess\w*|diagnos\w*|test\w*|treat\w*|manag\w*)\b.{0,100}\b(?:suspect\w*|probable|confirmed|patients?|cases?)\b",
        r"\b(?:guidance|recommendations?|protocol|instructions?)\b.{0,100}\b(?:treatment|treating|clinical management|therapeutics?)\b.{0,100}\b(?:ebola|marburg|filovirus|patients?|cases?)\b",
        r"\b(?:assess\w*|diagnos\w*|test\w*|treat\w*|manag\w*)\b.{0,100}\b(?:suspect\w*|probable|confirmed)\s+(?:ebola|marburg|cases?|patients?)\b",
        r"\bhow\s+to\s+handle\b.{0,100}\b(?:suspect\w*|probable|confirmed|ebola|marburg|patients?|cases?)\b",
        r"\b(?:my|our)\s+(?:child|patient|relative|family member)\b",
        r"\bi\s+(?:have|am experiencing|think i have)\b.{0,80}\b(?:ebola|fever|bleeding|vomiting|diarrhoea|diarrhea)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)
