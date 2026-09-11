"""Re-render reports/eval_summary.md from an already-saved run artifact under reports/runs/ —
no LLM call, no vector store, no API key. Deliberately its own module rather than a flag on
run_release.main(): that function's only other job is kicking off a real 30-item paid holdout run,
and a free re-render has no business sharing an argv namespace with that.

    python -m evals.render_summary reports/runs/holdout-0cf5e0821ec7.json

Existing evidence never moves without a reason: this refuses to render unless the artifact's own
recomputed structural metrics agree with what's stored in it, which is the check that a hand-edited
or corrupted artifact can't silently become published evidence.
"""

import argparse
import dataclasses
import json
import os
import tempfile
from pathlib import Path

from evals.run_release import (
    REPORTS_DIR,
    ReleaseRun,
    _write_summary_report,
    compute_structural_metrics,
)

_RUN_FIELDS = {f.name for f in dataclasses.fields(ReleaseRun)}


class ArtifactSchemaError(ValueError):
    """Raised when a run artifact's top-level keys don't match ReleaseRun's fields exactly."""


class StaleMetricsError(ValueError):
    """Raised when an artifact's stored structural_metrics disagree with what its own items
    recompute to — the artifact may have been hand-edited or corrupted."""


def load_run(path: Path) -> ReleaseRun:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_keys = set(raw)
    missing = _RUN_FIELDS - raw_keys
    unexpected = raw_keys - _RUN_FIELDS
    if missing or unexpected:
        raise ArtifactSchemaError(
            f"{path} does not match ReleaseRun's fields — missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return ReleaseRun(**raw)


def render(run: ReleaseRun, path: Path) -> None:
    recomputed = compute_structural_metrics(run.items)
    if recomputed != run.structural_metrics:
        raise StaleMetricsError(
            f"run {run.run_id}: stored structural_metrics do not match what run.items "
            "recompute to — refusing to publish a possibly hand-edited artifact as evidence"
        )
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _write_summary_report(run, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="path to a reports/runs/<run_id>.json file")
    args = parser.parse_args()

    run = load_run(args.artifact)
    summary_path = REPORTS_DIR / "eval_summary.md"
    render(run, summary_path)
    print(
        f"wrote {summary_path} from {args.artifact} "
        f"(run_id={run.run_id}, split={run.split}, label={run.label}, status={run.status})"
    )


if __name__ == "__main__":
    main()
