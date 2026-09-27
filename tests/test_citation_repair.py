from src import rag
from src.llm import LLMResponse
from src.retrieve import RetrievedChunk

CHUNK = RetrievedChunk(
    chunk_id="h", doc_id="sarb_d3_2023", text="t", page_start=3, page_end=3, section="", score=1.0
)


def _responses(monkeypatch, *texts):
    calls = []

    def fake(system, user, max_tokens=1024):
        calls.append(user)
        return LLMResponse(
            text=texts[len(calls) - 1],
            model="fake",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
        )

    monkeypatch.setattr(rag, "complete", fake)
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5, rerank=False: [CHUNK])
    return calls


def test_malformed_citation_is_repaired_once(monkeypatch):
    monkeypatch.setattr(rag, "CITATION_REPAIR", True)
    calls = _responses(
        monkeypatch,
        "Banks must comply (sarb_d3_2023 page 3)",
        "Banks must comply. [sarb_d3_2023, p.3]",
    )
    result = rag.answer_question("q?")
    assert not result.refused and result.repair_attempted and len(calls) == 2
    assert result.llm_response.cost_usd == 0.002
    assert result.first_raw_output.startswith("Banks must comply (")


def test_fabricated_citation_is_not_repaired(monkeypatch):
    monkeypatch.setattr(rag, "CITATION_REPAIR", True)
    calls = _responses(monkeypatch, "Banks must comply. [sarb_d3_2023, p.9]")
    result = rag.answer_question("q?")
    assert result.refused and result.refusal_reason == rag.RefusalReason.UNVERIFIED_CITATION
    assert len(calls) == 1 and not result.repair_attempted


def test_failed_repair_still_refuses(monkeypatch):
    monkeypatch.setattr(rag, "CITATION_REPAIR", True)
    _responses(monkeypatch, "Banks must comply.", "Still no citation.")
    result = rag.answer_question("q?")
    assert result.refused and result.repair_attempted


def test_repair_off_by_setting(monkeypatch):
    monkeypatch.setattr(rag, "CITATION_REPAIR", False)
    calls = _responses(monkeypatch, "Banks must comply.")
    assert rag.answer_question("q?").refused and len(calls) == 1
