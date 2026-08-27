"""agent.py against a mocked LLM and mocked retrieval — no API key, no vector store needed."""

from src import agent
from src.llm import LLMResponse
from src.retrieve import RetrievedChunk


def _chunk(chunk_id, doc_id, page=1, text="some regulatory text"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        page_start=page,
        page_end=page,
        section="",
        score=0.9,
    )


def _llm_queue(responses):
    calls = iter(responses)

    def fake_complete(system, user, max_tokens=1024):
        text = next(calls)
        return LLMResponse(
            text=text, model="fake", input_tokens=10, output_tokens=10, cost_usd=0.001
        )

    return fake_complete


def test_sufficient_context_answers_after_a_single_retrieval(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda q, k=5, rerank=False: [_chunk("c1", "doc_a")])
    monkeypatch.setattr(agent, "complete", _llm_queue(["SUFFICIENT", "The answer. [doc_a, p.1]"]))

    result = agent.answer_question("A question")

    assert len(result.steps) == 2  # retrieve, answer — no requery
    assert result.steps[0].action == "retrieve"
    assert result.steps[1].action == "answer"
    assert result.citations == [agent.Citation(doc_id="doc_a", page=1, verified=True)]


def test_insufficient_context_triggers_exactly_one_requery(monkeypatch):
    calls = {"n": 0}

    def fake_retrieve(q, k=5, rerank=False):
        calls["n"] += 1
        return [_chunk("c1", "doc_a")] if calls["n"] == 1 else [_chunk("c2", "doc_b")]

    monkeypatch.setattr(agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        agent,
        "complete",
        _llm_queue(["INSUFFICIENT: doc_b specific topic", "SUFFICIENT", "Answer. [doc_b, p.1]"]),
    )

    result = agent.answer_question("Compare doc_a and doc_b")

    actions = [s.action for s in result.steps]
    assert actions == ["retrieve", "requery", "answer"]
    assert result.steps[1].query == "doc_b specific topic"
    assert result.steps[1].new_chunk_ids == ["c2"]
    # both chunks are in the merged context by the time the final answer is generated
    assert {c.chunk_id for c in result.retrieved_chunks} == {"c1", "c2"}


def test_requerying_never_exceeds_max_steps(monkeypatch):
    # every decision says INSUFFICIENT — the loop must still terminate and answer, not spin
    monkeypatch.setattr(agent, "retrieve", lambda q, k=5, rerank=False: [_chunk("c1", "doc_a")])
    monkeypatch.setattr(
        agent,
        "complete",
        _llm_queue(
            [
                "INSUFFICIENT: more on doc_a",
                "INSUFFICIENT: even more",  # both consumed — MAX_STEPS-1 decision points always
                "Final answer regardless. [doc_a, p.1]",  # run; the loop just stops asking after
            ]
        ),
    )

    result = agent.answer_question("A question")

    actions = [s.action for s in result.steps]
    assert actions.count("requery") == agent.MAX_STEPS - 1
    assert actions[-1] == "answer"


def test_a_repeated_requery_does_not_duplicate_already_retrieved_chunks(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda q, k=5, rerank=False: [_chunk("c1", "doc_a")])
    monkeypatch.setattr(
        agent,
        "complete",
        _llm_queue(["INSUFFICIENT: same thing again", "SUFFICIENT", "Answer. [doc_a, p.1]"]),
    )

    result = agent.answer_question("A question")

    assert [c.chunk_id for c in result.retrieved_chunks] == ["c1"]
    assert result.steps[1].new_chunk_ids == []  # nothing new to report


def test_no_retrieved_chunks_refuses_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda q, k=5, rerank=False: [])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("complete() should not be called when retrieval returns nothing")

    monkeypatch.setattr(agent, "complete", fail_if_called)

    result = agent.answer_question("An unanswerable question")

    assert result.refused is True
    assert result.answer == agent.REFUSAL_PHRASE


def test_total_cost_sums_decision_calls_and_the_final_answer(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda q, k=5, rerank=False: [_chunk("c1", "doc_a")])
    monkeypatch.setattr(
        agent, "complete", _llm_queue(["INSUFFICIENT: more", "SUFFICIENT", "Answer."])
    )

    result = agent.answer_question("A question")

    # 2 decision calls (0.001 each) + 1 final answer call (0.001)
    assert result.total_cost_usd == 0.003
