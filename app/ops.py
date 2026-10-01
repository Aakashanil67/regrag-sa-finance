"""Usage dashboard. Reads live metrics from the API and saved evaluation reports locally.

HTTP access keeps the UI independent of the vector store and model packages. The API controls
which query fields are exposed.

    streamlit run app/ops.py"""

import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.ops_history import prepare_eval_history  # noqa: E402
from src.config import REPORTS_DIR  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RegRAG — Ops", page_icon="📊", layout="wide")
st.title("RegRAG Ops Dashboard")

st.header("Eval score history")
eval_history_path = REPORTS_DIR / "eval_runs.csv"
if eval_history_path.exists():
    eval_df = pd.read_csv(eval_history_path, parse_dates=["timestamp"])
    eval_df, metric_cols = prepare_eval_history(eval_df)
    if "evidence_scope" in eval_df:
        st.caption("Development and test observations are labelled separately.")
    st.line_chart(eval_df.set_index("timestamp")[metric_cols])
    st.dataframe(eval_df.tail(10), use_container_width=True)
else:
    st.info("No eval runs yet — run `python -m evals.run_eval --split dev --label ...`.")

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
            # question is null by default (see src/obslog.py), showing the column would just be
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
    st.info("No retrieval benchmark report yet.")
