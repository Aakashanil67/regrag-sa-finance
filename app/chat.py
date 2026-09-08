"""Streamlit chat UI. Talks to the FastAPI backend over HTTP — it never imports src.rag directly
— so the UI and the API stay two genuinely separate deployable pieces, matching the
docker-compose split (Phase 11) rather than a UI that happens to also contain the RAG logic.

    streamlit run app/chat.py
"""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Plain-language versions of src.rag.RefusalReason — never the raw model text, which the API
# already discards before this UI ever sees it.
_REFUSAL_REASON_LABELS = {
    "no_context": "Refused because nothing relevant was retrieved from the corpus.",
    "model_refusal": "The model reported it has no source for this question.",
    "malformed_refusal": "Refused because the model's response didn't match the expected refusal format.",
    "missing_citation": "Refused because the generated answer had no citation.",
    "uncited_line": "Refused because part of the generated answer had no citation.",
    "unverified_citation": "Refused because the generated citation did not match a retrieved page.",
}

st.set_page_config(page_title="RegRAG — SA Financial Regulation Assistant", page_icon="⚖️")
st.title("RegRAG")
st.caption(
    "Answers South African financial regulation questions from a fixed local corpus (SARB, "
    "IFRS 9, National Credit Act, FSCA). **Educational tool — not legal advice.**"
)

if "history" not in st.session_state:
    st.session_state.history = []


def render_source_notices(notices: list[dict]) -> None:
    # deterministic, code-generated disclosures (third-party source, withdrawn/superseded/draft
    # status — see src/rag.py's _source_notices) rendered separately from the model's own answer
    # text, on purpose: they're never part of what the eval harness grades as "the answer".
    # Ordinary historical context (e.g. a dated snapshot) uses info styling; withdrawn/superseded/
    # non-final sources get warning styling — a strong colour reserved for the cases that matter.
    strong_warning_kinds = {"withdrawn_source", "superseded_source", "non_final_source"}
    for notice in notices:
        label = "Source status: " + notice["text"]
        if notice["evidence"]:
            refs = ", ".join(f"[{e['doc_id']}, p.{e['page']}]" for e in notice["evidence"])
            label += f" (evidence: {refs})"
        if notice["kind"] in strong_warning_kinds:
            st.warning(label)
        else:
            st.info(label)


def render_sources(citations: list[dict], chunks: list[dict]) -> None:
    if citations:
        labels = ", ".join(f"[{c['doc_id']}, p.{c['page']}]" for c in citations)
        st.caption(f"Citations: {labels}")
        unverified = [c for c in citations if not c["verified"]]
        if unverified:
            st.warning(
                f"{len(unverified)} citation(s) point to a page that wasn't actually "
                "retrieved — the model may have invented them."
            )

    with st.expander(f"What was retrieved ({len(chunks)} chunks)"):
        for chunk in chunks:
            st.markdown(
                f"**{chunk['doc_id']}**, p.{chunk['page_start']}-{chunk['page_end']} "
                f"({chunk['section'] or 'no section detected'}) — score {chunk['score']:.2f}"
            )
            st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and turn.get("chunks"):
            render_source_notices(turn.get("source_notices", []))
            render_sources(turn["citations"], turn["chunks"])

question = st.chat_input("Ask about SARB, IFRS 9, the National Credit Act, or FSCA rules...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            response = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=60.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Couldn't reach the API at {API_URL}: {exc}")
        else:
            st.write(data["answer"])
            if data["refused"] and data.get("refusal_reason"):
                st.caption(
                    _REFUSAL_REASON_LABELS.get(data["refusal_reason"], data["refusal_reason"])
                )
            render_source_notices(data.get("source_notices", []))
            render_sources(data["citations"], data["retrieved_chunks"])
            st.caption(f"{data['latency_ms']:.0f} ms · ${data['cost_usd']:.4f} · {data['model']}")

            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "citations": data["citations"],
                    "chunks": data["retrieved_chunks"],
                    "source_notices": data.get("source_notices", []),
                }
            )
