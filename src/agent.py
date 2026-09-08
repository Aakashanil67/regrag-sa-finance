"""Bounded retrieve-decide-requery loop, built to test one specific, already-diagnosed failure
class rather than "agents are generally better."

reports/failure_analysis.md found that a two-document comparison question ("what do X and Y have
in common") embeds as a single query, which under-retrieves whichever named document has fewer
chunks — the larger document's vocabulary dominates the top-k regardless of relevance (g37, g44).
An agent that looks at what it retrieved, notices a named document is missing, and issues a
second, targeted retrieval call before answering should fix that directly. This module measures
whether it actually does, against plain single-shot RAG on the same questions
(scripts/agent_eval.py), rather than assuming an agentic loop is an improvement by construction —
extra LLM calls are extra latency and cost that need to earn their place.

Deliberately not built on the Anthropic SDK's tool-use API: the decision step needs exactly one
choice (answer now, or search once more for something specific), which a one-line structured
text response answers as reliably as a tool-call schema would, without pulling in a second calling
convention alongside rag.py's plain completions.
"""

from dataclasses import dataclass, field

from src.llm import LLMResponse, complete
from src.rag import (
    _SYSTEM_PROMPT,
    Citation,
    RefusalReason,
    SourceNotice,
    _format_context,
    _source_notices,
    validate_generated_answer,
)
from src.rag import INSUFFICIENT_CONTEXT_PHRASE as REFUSAL_PHRASE
from src.retrieve import RetrievedChunk, retrieve

MAX_STEPS = 3  # 1 initial retrieval + at most 2 requeries, before being forced to answer

_DECISION_PROMPT = """You are deciding whether you have enough information to answer a question \
about South African financial regulation, or whether you need to search for more.

Question: {question}

Context retrieved so far:
{context}

If the context above is enough to answer the question, respond with exactly:
SUFFICIENT

If the question compares or combines information from a specific named document that ISN'T \
represented in the context above, respond with exactly:
INSUFFICIENT: <a short, specific search query naming that missing document or topic>

Respond with nothing else."""


@dataclass
class AgentStep:
    action: str  # "retrieve" | "requery" | "answer"
    query: str
    new_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    refused: bool
    steps: list[AgentStep]
    decision_calls: list[LLMResponse]
    llm_response: LLMResponse  # the final answer-generating call, same shape as RAGResult's
    source_notices: list[SourceNotice] = field(default_factory=list)
    refusal_reason: RefusalReason | None = None

    @property
    def total_cost_usd(self) -> float:
        return self.llm_response.cost_usd + sum(c.cost_usd for c in self.decision_calls)


def _merge_new(
    accumulated: list[RetrievedChunk], new: list[RetrievedChunk]
) -> tuple[list[RetrievedChunk], list[str]]:
    seen = {c.chunk_id for c in accumulated}
    added = [c for c in new if c.chunk_id not in seen]
    return accumulated + added, [c.chunk_id for c in added]


def answer_question(question: str, k: int = 5) -> AgentResult:
    chunks = retrieve(question, k=k, rerank=True)
    steps = [
        AgentStep(action="retrieve", query=question, new_chunk_ids=[c.chunk_id for c in chunks])
    ]
    decision_calls: list[LLMResponse] = []

    for _ in range(MAX_STEPS - 1):
        if not chunks:
            break
        decision = complete(
            system="You are a careful research assistant that only searches for what's actually "
            "missing, never pads context speculatively.",
            user=_DECISION_PROMPT.format(question=question, context=_format_context(chunks)),
            max_tokens=128,
        )
        decision_calls.append(decision)
        text = decision.text.strip()
        if not text.upper().startswith("INSUFFICIENT"):
            break

        followup = text.split(":", 1)[1].strip() if ":" in text else question
        new_chunks = retrieve(followup, k=k, rerank=True)
        chunks, added_ids = _merge_new(chunks, new_chunks)
        steps.append(AgentStep(action="requery", query=followup, new_chunk_ids=added_ids))

    if not chunks:
        return AgentResult(
            question=question,
            answer=REFUSAL_PHRASE,
            citations=[],
            retrieved_chunks=[],
            refused=True,
            steps=steps,
            decision_calls=decision_calls,
            llm_response=LLMResponse(
                text="", model="none", input_tokens=0, output_tokens=0, cost_usd=0.0
            ),
            refusal_reason=RefusalReason.NO_CONTEXT,
        )

    steps.append(AgentStep(action="answer", query=question))
    user_message = f"{_format_context(chunks)}\n\nQuestion: {question}"
    llm_response = complete(system=_SYSTEM_PROMPT, user=user_message)

    validated = validate_generated_answer(llm_response.text, chunks)
    notices = [] if validated.refused else _source_notices(validated.citations)

    return AgentResult(
        question=question,
        answer=validated.answer,
        citations=validated.citations,
        retrieved_chunks=chunks,
        refused=validated.refused,
        steps=steps,
        decision_calls=decision_calls,
        llm_response=llm_response,
        source_notices=notices,
        refusal_reason=validated.refusal_reason,
    )
