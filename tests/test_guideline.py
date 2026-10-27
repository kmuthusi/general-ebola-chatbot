import pytest

import model_warmup
import simple_rag
from bounded_composer import REFUSAL
from conversation import clarification_for, contextualize_question
from public_safety import requires_professional_care
from simple_rag import SimpleAssistantRAG, SimpleAssistantRequest


def test_warmup_loads_embedding_then_chat_model(monkeypatch):
    calls = []
    monkeypatch.setattr(model_warmup.ollama, "embed", lambda **kwargs: calls.append(("embed", kwargs)))
    monkeypatch.setattr(model_warmup.ollama, "generate", lambda **kwargs: calls.append(("generate", kwargs)))
    model_warmup.warm_models("llama3.1:8b", "nomic-embed-text:latest")
    assert [name for name, _ in calls] == ["embed", "generate"]
    assert calls[1][1]["prompt"] == ""
    assert all(args["keep_alive"] == model_warmup.KEEP_ALIVE for _, args in calls)


@pytest.mark.parametrize("question", [
    "How should I manage an Ebola patient?",
    "How to administer mAb114?",
    "What dose of medicine should I use?",
    "What do the WHO guidelines say about IV fluid dosing?",
    "My child has fever after travel.",
    "I have fever and think I have Ebola.",
])
def test_public_care_requests_are_redirected(question):
    assert requires_professional_care(question)


def test_general_document_question_is_available_to_public():
    assert not requires_professional_care("What topics do the WHO guidelines cover?")


def test_follow_up_uses_only_recent_user_questions_for_context():
    history = ("What is Ebola?", "How does it spread?")
    assert contextualize_question("How is it transmitted?", history) == "How is Ebola transmitted?"
    assert "Which aspect of Ebola in children" in clarification_for("What about children?", history)
    assert contextualize_question("What about Marburg?", history) == (
        "What do the WHO guidelines say about Marburg disease?"
    )
    assert "compare Marburg with Ebola" in clarification_for("What about Marburg?", history)


def test_ambiguous_follow_up_asks_before_retrieval(monkeypatch):
    class FakeStore:
        def __init__(self, **kwargs):
            pass

        def ensure_index(self, pages):
            pytest.fail("Ambiguous follow-up should not reach retrieval")

    monkeypatch.setattr(simple_rag, "VectorStore", FakeStore)
    rag = SimpleAssistantRAG("who_guidelines")
    result = rag.ask(SimpleAssistantRequest(
        "What about children?", earlier_questions=("What is Ebola?",)
    ))
    assert "Which aspect of Ebola in children" in result["answer"]
    assert result["chunks"] == []


def test_empty_question_is_rejected(monkeypatch):
    monkeypatch.setattr(simple_rag, "VectorStore", lambda **kwargs: object())
    rag = SimpleAssistantRAG("who_guidelines")
    with pytest.raises(ValueError):
        rag.ask(SimpleAssistantRequest(" "))


def test_refusal_remains_available():
    assert "WHO documents" in REFUSAL
