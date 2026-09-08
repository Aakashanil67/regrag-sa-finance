"""Observability dashboard: eval-score history, query volume/latency/cost, retrieval benchmark.

Usage data comes from the API over HTTP (`/stats`, `/recent-queries`), not by importing
src.obslog or opening the SQLite log directly — this is the boundary that lets the UI image drop
Chroma, Torch, sentence-transformers, and the Anthropic/eval packages entirely (see
requirements-ui.txt), and it means this dashboard can never see more than the API is willing to
serve, including the privacy default that keeps raw question text out of the response. Eval
history and the retrieval benchmark are still read directly from the report files the eval
pipeline writes, since those are static local artefacts, not live query data.

    streamlit run app/ops.py
"""

import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RegRAG — Ops", page_icon="📊", layout="wide")
st.title("RegRAG Ops Dashboard")

st.header("Eval score history")
eval_history_path = REPORTS_DIR / "eval_history.csv"
if eval_history_path.exists():
    eval_df = pd.read_csv(eval_history_path, parse_dates=["timestamp"])
    metric_cols = [c for c in eval_df.columns if c != "timestamp"]
    st.line_chart(eval_df.set_index("timestamp")[metric_cols])
    st.dataframe(eval_df.tail(10), use_container_width=True)
else:
    st.info("No eval runs yet — run `python -m evals.run_release --split dev --label ...`.")

st.header("Usage")
try:
    stats = httpx.get(f"{API_URL}/stats", timeout=10.0).raise_for_status().json()
except httpx.HTTPError as exc:
    st.error(f"Couldn't reach the API at {API_URL}: {exc}")
    stats = None

if stats is not None:
    raw_logging_on = stats["content_logging_enabled"]
    st.caption(
        f"Raw content logging: {'**on**' if raw_logging_on else '**off**'} (`LOG_RAW_CONTENT`)"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total queries", stats["total_queries"])
    col2.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
    col3.metric("Total cost", f"${stats['total_cost_usd']:.4f}")
    col4.metric("Refusal rate", f"{stats['refusal_rate']:.0%}")

    queries = httpx.get(f"{API_URL}/recent-queries", params={"limit": 100}, timeout=10.0).json()
    if queries:
        queries_df = pd.DataFrame(queries)
        queries_df["timestamp"] = pd.to_datetime(queries_df["timestamp"], unit="s")
        st.line_chart(queries_df.set_index("timestamp")[["latency_ms", "cost_usd"]])
        if raw_logging_on:
            columns = [
                "timestamp",
                "question",
                "refused",
                "citation_count",
                "latency_ms",
                "cost_usd",
            ]
        else:
            # question is null by default (see src/obslog.py) — showing the column would just be
            # a column of blanks, which reads as a bug rather than a deliberate privacy default
            columns = ["timestamp", "refused", "citation_count", "latency_ms", "cost_usd"]
        st.dataframe(queries_df[columns], use_container_width=True)
    else:
        st.info("No queries logged yet — ask something in the chat UI first.")

st.header("Retrieval benchmark")
bench_path = REPORTS_DIR / "retrieval_bench.md"
if bench_path.exists():
    st.markdown(bench_path.read_text(encoding="utf-8"))
else:
    st.info("No retrieval benchmark yet — run `python -m evals.retrieval_bench`.")
