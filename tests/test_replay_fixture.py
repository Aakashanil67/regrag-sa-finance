import json
from pathlib import Path

import pytest

from src import rag
from src.retrieve import RetrievedChunk

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "replay.json").read_text(encoding="utf-8")
)


def _chunks(contexts):
    return [
        RetrievedChunk(
            chunk_id=c[0],
            doc_id=c[1],
            text=c[5],
            page_start=c[2],
            page_end=c[3],
            section=c[4],
            score=0.0,
        )
        for c in contexts
    ]


@pytest.mark.parametrize("item", FIXTURE["items"], ids=lambda i: i["id"])
def test_validator_reproduces_recorded_outcome(item):
    result = rag.validate_generated_answer(item["raw_model_output"], _chunks(item["contexts"]))
    assert result.refused == item["refused"]
    reason = result.refusal_reason.value if result.refusal_reason else None
    assert reason == item["refusal_reason"]
    got = [(c.doc_id, c.page, c.verified) for c in result.citations]
    want = [(c["doc_id"], c["page"], c["verified"]) for c in item["citations"]]
    assert got == want
