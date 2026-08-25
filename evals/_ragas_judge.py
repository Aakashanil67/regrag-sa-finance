"""Shared RAGAS judge setup — used by both run_ragas.py and record_fixtures.py so the Anthropic
top_p workaround (see below) lives in exactly one place."""

from anthropic import AsyncAnthropic
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory

from src.config import DEFAULT_ANTHROPIC_MODEL, EMBEDDING_MODEL_NAME


def build_judge():
    client = AsyncAnthropic()
    # max_tokens=4096: the default 1024 truncates mid-JSON on an answer with many claims to
    # verify (faithfulness's NLI step lists a verdict per statement) — a real failure hit on the
    # first live run, "EOF while parsing a list", not a hypothetical worth guarding against.
    llm = llm_factory(DEFAULT_ANTHROPIC_MODEL, provider="anthropic", client=client, max_tokens=4096)
    # ragas's instructor adapter hardcodes both temperature and top_p for every provider. Claude
    # rejects a request that sets both, and rejects an explicit top_p=None just as strictly
    # ("Input should be a valid number") — the key has to be absent from the request, not nulled.
    del llm.model_args["top_p"]
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME)
    return llm, embeddings
