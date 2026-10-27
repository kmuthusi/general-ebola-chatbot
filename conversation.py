"""Resolve short follow-up questions without trusting prior generated answers."""

import re


TOPIC_RE = re.compile(r"\b(ebola|marburg|bundibugyo|sudan|filovirus)\b", re.I)
TOPIC_ONLY_RE = re.compile(
    r"(?:(?:and|what|how)\s+about)\s+(ebola|marburg|bundibugyo|sudan|filovirus)\??",
    re.I,
)
POPULATION_RE = re.compile(
    r"^(?:(?:and|what|how)\s+about|and\s+in)\s+"
    r"(children|child|kids|infants|babies|pregnancy|pregnant women)\??$",
    re.I,
)


def recent_topic(earlier_questions: tuple[str, ...]) -> str | None:
    for question in reversed(earlier_questions[-3:]):
        match = TOPIC_RE.search(question)
        if match:
            return match.group(1).capitalize()
    return None


def clarification_for(question: str, earlier_questions: tuple[str, ...] = ()) -> str | None:
    current = question.strip()
    topic_match = TOPIC_ONLY_RE.fullmatch(current)
    if topic_match:
        topic = topic_match.group(1).capitalize()
        previous = recent_topic(earlier_questions)
        if previous and previous.lower() != topic.lower():
            return (
                f"Would you like to compare {topic} with {previous}, or ask about "
                f"{topic}'s symptoms, transmission, or coverage in the WHO guidelines?"
            )
        return f"Which aspect of {topic} do you mean—symptoms, transmission, or care?"
    match = POPULATION_RE.fullmatch(current)
    if not match:
        return None
    population = match.group(1).lower()
    topic = recent_topic(earlier_questions) or "filovirus disease"
    return (
        f"Which aspect of {topic} in {population} do you mean—symptoms, "
        "exposure, or care? Please ask a specific question so I can use the right section."
    )


def contextualize_question(question: str, earlier_questions: tuple[str, ...] = ()) -> str:
    current = question.strip()
    if earlier_questions:
        aspect = re.fullmatch(
            r"(symptoms?|transmission|exposure|care|treatment|risk|prevention|coverage|comparison|compare)\??",
            current,
            re.I,
        )
        if aspect:
            last = earlier_questions[-1].strip()
            population = POPULATION_RE.fullmatch(last)
            if population:
                topic = recent_topic(earlier_questions[:-1]) or "filovirus disease"
                return (
                    f"What do the WHO guidelines say about {aspect.group(1).lower()} "
                    f"of {topic} in {population.group(1).lower()}?"
                )
            named_topic = TOPIC_ONLY_RE.fullmatch(last)
            if named_topic:
                topic = named_topic.group(1).capitalize()
                if aspect.group(1).lower() in {"comparison", "compare"}:
                    previous = recent_topic(earlier_questions[:-1])
                    if previous:
                        return f"How do the WHO guidelines compare {topic} with {previous}?"
                return f"What do the WHO guidelines say about {aspect.group(1).lower()} of {topic} disease?"
    topic_only = TOPIC_ONLY_RE.fullmatch(current)
    if topic_only:
        return f"What do the WHO guidelines say about {topic_only.group(1).capitalize()} disease?"
    if TOPIC_RE.search(current) or not earlier_questions:
        return current
    follow_up = re.search(r"\b(it|its|they|them|this|that|those|these|same)\b", current, re.I)
    follow_up = follow_up or re.match(r"^(and|also|what about|how about)\b", current, re.I)
    if not follow_up:
        return current
    topic = recent_topic(earlier_questions)
    if topic:
        if re.match(r"^(what|how) about\b", current, re.I):
            detail = re.sub(r"^(what|how) about\s+", "", current, flags=re.I).rstrip(" ?")
            return f"What do the WHO guidelines say about {detail} in {topic} disease?"
        resolved = re.sub(r"\b(it|its|they|them|this|that|those|these)\b", topic,
                          current, count=1, flags=re.I)
        if resolved != current:
            return resolved
        return f"{current}\nDisease in question: {topic}"
    prior = [item.strip() for item in earlier_questions[-2:] if item.strip()]
    if not prior:
        return current
    return f"{current}\nEarlier user question for context only: {'; '.join(prior)}"
