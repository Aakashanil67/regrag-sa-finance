"""Observability dashboard: eval-score history, query volume/latency/cost, retrieval benchmark.

Reads directly from the files the eval and logging pipelines already write —
reports/eval_history.csv, the SQLite query log, reports/retrieval_bench.md — rather than
recomputing anything, so this page stays honest about "what actually happened" instead of
re-deriving numbers that could drift from the reports a reader might cross-check it against.

    streamlit run app/ops.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR  # noqa: E402
from src.obslog import recent_queries, stats_summary  # noqa: E402

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
    st.info("No eval runs yet — run `python -m evals.run_ragas` to populate this chart.")

st.header("Usage")
stats = stats_summary()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries", stats["total_queries"])
col2.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
col3.metric("Total cost", f"${stats['total_cost_usd']:.4f}")
col4.metric("Refusal rate", f"{stats['refusal_rate']:.0%}")

queries = recent_queries(limit=100)
if queries:
    queries_df = pd.DataFrame(queries)
    queries_df["timestamp"] = pd.to_datetime(queries_df["timestamp"], unit="s")
    st.line_chart(queries_df.set_index("timestamp")[["latency_ms", "cost_usd"]])
    st.dataframe(
        queries_df[
            ["timestamp", "question", "refused", "citation_count", "latency_ms", "cost_usd"]
        ],
        use_container_width=True,
    )
else:
    st.info("No queries logged yet — ask something in the chat UI first.")

st.header("Retrieval benchmark")
bench_path = REPORTS_DIR / "retrieval_bench.md"
if bench_path.exists():
    st.markdown(bench_path.read_text(encoding="utf-8"))
else:
    st.info("No retrieval benchmark yet — run `python -m evals.retrieval_bench`.")
