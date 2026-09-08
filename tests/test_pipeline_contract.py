"""rag.py and agent.py must fail closed identically — both call src.rag.validate_generated_answer,
so a citation contract violation can't slip through one path just because it was fixed in the
other. These tests exercise validate_generated_answer directly, then confirm each pipeline surface
actually delegates to it rather than re-implementing its own (drifting) copy of the same rules.
"""

from src import agent, rag
from src.retrieve import RetrievedChunk


def _chunk(doc_id="sarb_d3_2023", page_start=1, page_end=1, text="some regulatory text"):
    return RetrievedChunk(
        chunk_id="hash1",
        doc_id=doc_id,
        text=text,
        page_start=page_start,
        page_end=page_end,
        section="",
        score=0.9,
    )


def test_validator_accepts_a_fully_cited_single_line_answer():
    result = rag.validate_generated_answer("Banks must comply. [sarb_d3_2023, p.1]", [_chunk()])

    assert result.refused is False
    assert result.refusal_reason is None
    assert result.citations == [rag.Citation(doc_id="sarb_d3_2023", page=1, verified=True)]


def test_validator_refuses_exact_refusal_phrase_as_model_refusal():
    result = rag.validate_generated_answer(rag.INSUFFICIENT_CONTEXT_PHRASE, [_chunk()])

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.MODEL_REFUSAL
    assert result.answer == rag.INSUFFICIENT_CONTEXT_PHRASE


def test_validator_refuses_missing_citation():
    result = rag.validate_generated_answer("The Act requires disclosure.", [_chunk()])

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.MISSING_CITATION


def test_validator_refuses_uncited_line_even_with_a_cited_line_present():
    text = "First supported claim. [sarb_d3_2023, p.1]\nSecond unsupported claim."
    result = rag.validate_generated_answer(text, [_chunk()])

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.UNCITED_LINE


def test_validator_refuses_a_page_hallucinated_citation():
    text = "Banks must comply. [sarb_d3_2023, p.99]"
    result = rag.validate_generated_answer(text, [_chunk(page_start=3, page_end=3)])

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.UNVERIFIED_CITATION


def test_validator_refuses_malformed_refusal_text():
    text = f"{rag.INSUFFICIENT_CONTEXT_PHRASE} But outside knowledge says otherwise."
    result = rag.validate_generated_answer(text, [_chunk()])

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.MALFORMED_REFUSAL


def test_validator_refuses_a_citation_to_a_doc_id_never_retrieved():
    text = "Banks must comply. [never_retrieved_doc, p.1]"
    result = rag.validate_generated_answer(text, [_chunk(doc_id="sarb_d3_2023")])

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.UNVERIFIED_CITATION


def test_validator_accepts_a_verified_page_range_citation():
    text = "The rules apply broadly. [sarb_d3_2023, p.5-7]"
    result = rag.validate_generated_answer(text, [_chunk(page_start=5, page_end=7)])

    assert result.refused is False
    assert {c.page for c in result.citations} == {5, 6, 7}


def test_never_returns_the_raw_invalid_model_text_in_the_validated_answer():
    text = "Unverifiable claim with a hallucinated page. [sarb_d3_2023, p.999]"
    result = rag.validate_generated_answer(text, [_chunk(page_start=1, page_end=1)])

    assert text not in result.answer
    assert result.answer == rag.INSUFFICIENT_CONTEXT_PHRASE


def test_rag_no_retrieved_context_refuses_with_no_context_reason(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5, rerank=True: [])

    result = rag.answer_question("An unanswerable question")

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.NO_CONTEXT


def test_rag_valid_multi_document_answer_is_accepted(monkeypatch):
    text = "Both instruments address credit risk. [sarb_d3_2023, p.1] [sarb_d8_2023, p.2]"
    chunks = [
        _chunk(doc_id="sarb_d3_2023"),
        _chunk(doc_id="sarb_d8_2023", page_start=2, page_end=2),
    ]
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5, rerank=True: chunks)

    def fake_complete(system, user, max_tokens=1024):
        from src.llm import LLMResponse

        return LLMResponse(text=text, model="fake", input_tokens=1, output_tokens=1, cost_usd=0.0)

    monkeypatch.setattr(rag, "complete", fake_complete)

    result = rag.answer_question("Compare them")

    assert result.refused is False
    assert len(result.citations) == 2
    assert all(c.verified for c in result.citations)


def test_agent_final_answer_fails_closed_on_an_uncited_answer_even_after_a_requery(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda q, k=5, rerank=True: [_chunk()])
    monkeypatch.setattr(
        agent,
        "complete",
        _queue(["INSUFFICIENT: more detail", "SUFFICIENT", "Uncited final claim with no source."]),
    )

    result = agent.answer_question("A question")

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.MISSING_CITATION


def test_agent_final_answer_fails_closed_on_a_page_hallucinated_citation(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda q, k=5, rerank=True: [_chunk()])
    monkeypatch.setattr(
        agent,
        "complete",
        _queue(["SUFFICIENT", "Banks must comply. [sarb_d3_2023, p.999]"]),
    )

    result = agent.answer_question("A question")

    assert result.refused is True
    assert result.refusal_reason == rag.RefusalReason.UNVERIFIED_CITATION


def _queue(responses):
    from src.llm import LLMResponse

    calls = iter(responses)

    def fake_complete(system, user, max_tokens=1024):
        return LLMResponse(
            text=next(calls), model="fake", input_tokens=10, output_tokens=10, cost_usd=0.0
        )

    return fake_complete
