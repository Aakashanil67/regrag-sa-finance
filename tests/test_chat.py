from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_cached_history_keeps_withdrawn_notice():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app/chat.py"))
    app.session_state["history"] = [
        {
            "role": "assistant",
            "content": "Historical answer [old, p.1]",
            "citations": [{"doc_id": "old", "page": 1, "verified": True}],
            "chunks": [],
            "source_notices": [
                {
                    "kind": "withdrawn_source",
                    "text": "This source is withdrawn.",
                    "evidence": [],
                }
            ],
        }
    ]

    app.run()

    assert not app.exception
    assert any("withdrawn" in warning.value for warning in app.warning)
    app.run()
    assert any("withdrawn" in warning.value for warning in app.warning)


def test_history_replays_refusal_explanation_without_chunk_previews():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app/chat.py"))
    app.session_state["history"] = [
        {
            "role": "assistant",
            "content": "I don't have a source for that.",
            "citations": [],
            "chunks": [],
            "source_notices": [],
            "refused": True,
            "refusal_reason": "no_context",
        }
    ]

    app.run()

    assert not app.exception
    assert any("nothing relevant was retrieved" in caption.value for caption in app.caption)
