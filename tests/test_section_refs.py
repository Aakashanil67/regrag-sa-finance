from src import rag
from src.llm import LLMResponse
from src.retrieve import RetrievedChunk


def _chunk(doc_id, page, section):
    return RetrievedChunk(
        chunk_id=f"{doc_id}-{page}",
        doc_id=doc_id,
        text="text",
        page_start=page,
        page_end=page,
        section=section,
        score=1.0,
    )


def test_section_ref_from_numbered_heading():
    assert rag._section_ref("81. Prevention of reckless credit", "primary_legislation") == "s 81"
    assert rag._section_ref("8.1.2 Scope", "binding_regulatory_instrument") == "para 8.1.2"


def test_no_section_ref_for_unnumbered_or_year_headings():
    assert rag._section_ref("Executive summary", "primary_legislation") is None
    assert rag._section_ref("2023 amendments", "primary_legislation") is None
    assert rag._section_ref(None, None) is None


def test_verified_citation_carries_section_ref(monkeypatch):
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda q, k=5, rerank=False: [
            _chunk("nca_act_34_2005", 47, "81. Prevention of reckless credit")
        ],
    )
    monkeypatch.setattr(
        rag,
        "complete",
        lambda system, user, max_tokens=1024: LLMResponse(
            text="A credit provider must assess affordability. [nca_act_34_2005, p.47]",
            model="fake",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
        ),
    )
    result = rag.answer_question("What must a credit provider assess?")
    assert result.citations[0].section_ref == "s 81"
