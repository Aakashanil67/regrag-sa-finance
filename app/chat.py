"""Streamlit chat UI. Calls the FastAPI backend over HTTP.

streamlit run app/chat.py"""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Answered at build time in the hosted demo so they come straight from the response cache.
EXAMPLE_QUESTIONS = [
    "Which seven categories are the operational resilience principles organised under?",
    "What is a primary credit bureau in the context of disputed credit information?",
    "From which date did IFRS 9 take effect for annual reporting periods?",
    "Does IAS 39 still govern how financial instruments are accounted for now that IFRS 9 is final?",
    "On what grounds may a credit provider not unfairly discriminate when it assesses a person's ability to meet a credit agreement?",
]

# Plain-language versions of src.rag.RefusalReason, never the raw model text, which the API
# already discards before this UI ever sees it.
_REFUSAL_REASON_LABELS = {
    "no_context": "Refused because nothing relevant was retrieved from the corpus.",
    "model_refusal": "The model reported it has no source for this question.",
    "malformed_refusal": "Refused because the model's response didn't match the expected refusal format.",
    "missing_citation": "Refused because the generated answer had no citation.",
    "malformed_citation": (
        "Refused because the generated answer had a citation-like reference that didn't match "
        "the expected format."
    ),
    "uncited_line": "Refused because part of the generated answer had no citation.",
    "unverified_citation": "Refused because the generated citation did not match a retrieved page.",
}

st.set_page_config(page_title="RegRAG — SA Financial Regulation Assistant", page_icon="⚖️")
st.title("RegRAG")
st.caption(
    "Answers questions on South African banking and consumer-credit regulation (Prudential "
    "Authority directives, the Banks Act and its regulations, the National Credit Act and its "
    "regulations, FSCA conduct standards, IFRS 9) from a fixed corpus, citing the page. Not "
    "legal advice."
)

st.caption(
    "Answers come from GPT-5.6 Luna by default; set LLM_PROVIDER=ollama in .env to run a free "
    "local model instead."
)

if "history" not in st.session_state:
    st.session_state.history = []


def render_source_notices(notices: list[dict]) -> None:
    # deterministic, code-generated disclosures (third-party source, withdrawn/superseded/draft
    # status, see src/rag.py's _source_notices) rendered separately from the model's own answer
    # text, on purpose: they're never part of what the eval harness grades as "the answer".
    # Ordinary historical context (e.g. a dated snapshot) uses info styling. Withdrawn/superseded/
    # non-final sources get warning styling, a strong colour reserved for the cases that matter.
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


def render_refusal_explanation(refused: bool, refusal_reason: str | None) -> None:
    if refused and refusal_reason:
        st.caption(_REFUSAL_REASON_LABELS.get(refusal_reason, refusal_reason))


def render_sources(citations: list[dict], chunks: list[dict]) -> None:
    if citations:
        labels = []
        for citation in citations:
            title = citation.get("title") or citation["doc_id"]
            if citation.get("section_ref"):
                label = f"{title}, {citation['section_ref']}, p.{citation['page']}"
            else:
                label = f"{title}, p.{citation['page']}"
            if citation.get("source_url"):
                label = f"[{label}]({citation['source_url']})"
            else:
                label = f"`{label}`"
            labels.append(label)
        st.markdown(f"Citations: {', '.join(dict.fromkeys(labels))}")
        unverified = [c for c in citations if not c["verified"]]
        if unverified:
            st.warning(
                f"{len(unverified)} citation(s) point to a page that wasn't actually "
                "retrieved — the model may have invented them."
            )

    if not chunks:
        return
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
        if turn["role"] == "assistant":
            render_refusal_explanation(turn.get("refused", False), turn.get("refusal_reason"))
            render_source_notices(turn.get("source_notices", []))
            render_sources(turn.get("citations", []), turn.get("chunks", []))

clicked = None
for index, example in enumerate(EXAMPLE_QUESTIONS):
    if st.button(example, key=f"example_{index}"):
        clicked = example

question = (
    st.chat_input("Ask about SARB, IFRS 9, the National Credit Act, or FSCA rules...") or clicked
)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            response = httpx.post(f"{API_URL}/ask", json={"question": question}, timeout=300.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Couldn't reach the API at {API_URL}: {exc}")
        else:
            st.write(data["answer"])
            render_refusal_explanation(data["refused"], data.get("refusal_reason"))
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
                    "refused": data.get("refused", False),
                    "refusal_reason": data.get("refusal_reason"),
                }
            )
