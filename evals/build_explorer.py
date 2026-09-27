"""Build a static page for browsing the sealed test run: each question, the answer the frozen
pipeline served, the passages it cited, both judges' labels and closed-book GPT-5.6 Luna's answer.

    python -m evals.build_explorer   # writes deploy/explorer/index.html

The page is self-contained (no network requests), so any static host can serve it.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "reports" / "runs"
OUT = ROOT / "deploy" / "explorer" / "index.html"
JUDGES = ("qwen", "llama")
METRICS = ("answer_rate", "refusal_recall", "task_outcome")


def _load(name: str) -> dict:
    return json.loads((RUNS / name).read_text(encoding="utf-8"))


def _labels(label: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for judge in JUDGES:
        for row in _load(f"test-{label}.judge-{judge}.json")["items"]:
            out.setdefault(row["id"], {})[judge] = {
                "correctness": row.get("correctness"),
                "support": row.get("support"),
            }
    return out


def cited_passages(item: dict) -> list[dict]:
    passages, seen = [], set()
    for citation in item["citations"]:
        for ctx in item["contexts"]:
            covers = ctx["page_start"] <= citation["page"] <= ctx["page_end"]
            if ctx["doc_id"] == citation["doc_id"] and covers and ctx["chunk_id"] not in seen:
                seen.add(ctx["chunk_id"])
                passages.append(
                    {
                        "doc_id": ctx["doc_id"],
                        "pages": [ctx["page_start"], ctx["page_end"]],
                        "section": ctx["section"],
                        "text": ctx["text"],
                    }
                )
    return passages


def build_data() -> dict:
    final, closed = _load("test-final.json"), _load("test-closedbook.json")
    manifest = json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    titles = {e["id"]: e["title"] for e in manifest}
    final_labels, closed_labels = _labels("final"), _labels("closedbook")
    closed_items = {i["id"]: i for i in closed["items"]}
    items = []
    for it in final["items"]:
        cb = closed_items.get(it["id"], {})
        items.append(
            {
                "id": it["id"],
                "type": it["type"],
                "question": it["question"],
                "reference_answer": it.get("reference_answer") or "",
                "evidence": it.get("evidence") or [],
                "answer": it.get("served_answer"),
                "refused": it["refused"],
                "refusal_reason": it.get("refusal_reason"),
                "citations": it["citations"],
                "passages": cited_passages(it),
                "notices": [n["text"] for n in it.get("source_notices", [])],
                "judges": final_labels.get(it["id"], {}),
                "closed_book": {
                    "answer": cb.get("served_answer"),
                    "refused": cb.get("refused"),
                    "judges": closed_labels.get(it["id"], {}),
                },
            }
        )
    used = {c["doc_id"] for i in items for c in i["citations"]} | {
        e["doc_id"] for i in items for e in i["evidence"]
    }
    return {
        "run_date": final["created_at"][:10],
        "metrics": {k: final["metrics"][k] for k in METRICS},
        "closed_metrics": {k: closed["metrics"][k] for k in METRICS},
        "titles": {d: titles.get(d, d) for d in sorted(used)},
        "items": items,
    }


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(build_data()), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RegRAG test run explorer</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--ok:#1f7a4d;--bad:#b3261e;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29c;--line:#34332f;--card:#1f1f1d;--ok:#5cc38c;--bad:#f2857d;--warn:#e0b14f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px} p.lede{color:var(--muted);margin:0 0 16px}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.stat{border:1px solid var(--line);background:var(--card);border-radius:8px;padding:8px 12px;font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
select,input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
input{flex:1;min-width:180px}
details{border:1px solid var(--line);background:var(--card);border-radius:8px;margin-bottom:8px}
summary{cursor:pointer;padding:10px 12px;list-style:none;display:flex;gap:8px;align-items:baseline}
summary .id{color:var(--muted);font-size:12px;min-width:32px}
.tag{font-size:11px;border:1px solid var(--line);border-radius:10px;padding:0 6px;color:var(--muted)}
.body{padding:0 12px 12px;border-top:1px solid var(--line)}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:14px 0 4px}
.answer{white-space:pre-wrap} .refused{color:var(--warn)} .notice{color:var(--warn);font-size:13px}
blockquote{margin:6px 0;padding:6px 10px;border-left:3px solid var(--line);font-size:13px;white-space:pre-wrap;max-height:220px;overflow:auto}
.correct{color:var(--ok)} .incorrect,.unsupported,.answered_unanswerable{color:var(--bad)} .partial{color:var(--warn)}
</style></head><body><main>
<h1>RegRAG: sealed test run</h1>
<p class="lede" id="lede"></p>
<div class="stats" id="stats"></div>
<div class="controls">
<select id="type"><option value="">All question types</option></select>
<select id="outcome"><option value="">All outcomes</option><option value="answered">Answered</option><option value="refused">Refused</option><option value="wrong">Marked incorrect by a judge</option></select>
<input id="q" type="search" placeholder="Search questions and answers">
</div>
<div id="list"></div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const pct = r => r.n ? `${r.k}/${r.n}` : "n/a";
document.getElementById("lede").textContent = `All ${D.items.length} questions of the test set, run once on ${D.run_date} after the pipeline was frozen. Answers are from GPT-5.6 Luna with retrieval and citation checks; closed-book shows the same model without retrieval. Judges are local Qwen 2.5 7B and Llama 3.1 8B. No answer here has been checked by a person yet.`;
const stats = document.getElementById("stats");
[["Answerable questions answered", D.metrics.answer_rate], ["Unanswerable questions refused", D.metrics.refusal_recall], ["Closed-book: unanswerable refused", D.closed_metrics.refusal_recall]].forEach(([k, r]) => stats.append(el("div", "stat", `${k}: ${pct(r)}`)));
const typeSel = document.getElementById("type");
[...new Set(D.items.map(i => i.type))].sort().forEach(t => { const o = el("option", "", t.replace("_", " ")); o.value = t; typeSel.append(o); });
const labels = j => Object.entries(j || {}).map(([name, l]) => { const s = el("span"); s.append(`${name}: `); s.append(el("span", l.correctness || "", l.correctness || "-")); if (l.support) { s.append(", support "); s.append(el("span", l.support, l.support)); } return s; });
function card(i) {
  const d = el("details"); const s = el("summary");
  s.append(el("span", "id", i.id), el("span", "", i.question), el("span", "tag", i.type.replace("_", " ")), el("span", "tag", i.refused ? "refused" : "answered"));
  const b = el("div", "body");
  b.append(el("h3", "", "Pipeline answer"));
  b.append(i.refused ? el("div", "refused", `Refused (${i.refusal_reason || "no reason recorded"})`) : el("div", "answer", i.answer));
  i.notices.forEach(n => b.append(el("div", "notice", n)));
  if (i.citations.length) { b.append(el("h3", "", "Cited")); i.citations.forEach(c => b.append(el("div", "", `${D.titles[c.doc_id] || c.doc_id}${c.section_ref ? ", " + c.section_ref : ""}, p.${c.page}`))); }
  if (i.passages.length) { b.append(el("h3", "", "Passages the answer was checked against")); i.passages.forEach(p => { b.append(el("div", "tag", `${D.titles[p.doc_id] || p.doc_id} p.${p.pages[0]}${p.pages[1] !== p.pages[0] ? "-" + p.pages[1] : ""}`)); b.append(el("blockquote", "", p.text)); }); }
  b.append(el("h3", "", "Judges")); const jl = el("div"); labels(i.judges).forEach((s, n) => { if (n) jl.append("  ·  "); jl.append(s); }); b.append(jl);
  b.append(el("h3", "", "Reference answer")); b.append(el("div", "", i.reference_answer || "(unanswerable from the corpus)"));
  i.evidence.forEach(e => b.append(el("blockquote", "", `${D.titles[e.doc_id] || e.doc_id} p.${e.page}: "${e.quote}"`)));
  b.append(el("h3", "", "Closed-book GPT-5.6 Luna (no retrieval)"));
  b.append(i.closed_book.refused ? el("div", "refused", "Declined to answer") : el("div", "answer", i.closed_book.answer || "-"));
  const cl = el("div"); labels(i.closed_book.judges).forEach((s, n) => { if (n) cl.append("  ·  "); cl.append(s); }); b.append(cl);
  d.append(s, b); return d;
}
function draw() {
  const t = typeSel.value, o = document.getElementById("outcome").value, q = document.getElementById("q").value.toLowerCase();
  const list = document.getElementById("list"); list.replaceChildren();
  D.items.filter(i => (!t || i.type === t)
    && (!o || (o === "answered" && !i.refused) || (o === "refused" && i.refused) || (o === "wrong" && Object.values(i.judges).some(l => l.correctness === "incorrect")))
    && (!q || (i.question + " " + (i.answer || "")).toLowerCase().includes(q))).forEach(i => list.append(card(i)));
}
["type", "outcome", "q"].forEach(id => document.getElementById(id).addEventListener("input", draw));
draw();
</script></body></html>
"""


if __name__ == "__main__":
    main()
