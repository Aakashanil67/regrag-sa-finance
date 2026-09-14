import asyncio

import pytest

from evals import record_fixtures


def test_snapshot_budget_estimate_covers_generation_and_one_faithfulness_judge_per_item():
    estimate = record_fixtures.estimate_recording_cost(10, "claude-haiku-4-5")

    assert estimate == pytest.approx(10 * (0.013312 + 0.006656))


def test_historical_snapshot_path_is_rejected():
    with pytest.raises(ValueError, match="historical"):
        record_fixtures.validate_output_path(record_fixtures.FIXTURES_PATH)


def test_recording_requires_a_positive_cap_before_provider_construction(monkeypatch, tmp_path):
    called = False

    def fail_provider():
        nonlocal called
        called = True
        raise AssertionError("provider construction must not happen")

    monkeypatch.setattr(record_fixtures, "build_judge", fail_provider)

    with pytest.raises(ValueError, match="positive"):
        asyncio.run(
            record_fixtures.run(
                output_path=tmp_path / "ci_subset_v2.json",
                max_cost_usd=None,
            )
        )

    assert called is False


def test_recording_rejects_a_cap_below_the_full_estimate_before_provider_construction(
    monkeypatch, tmp_path
):
    called = False

    def fail_provider():
        nonlocal called
        called = True
        raise AssertionError("provider construction must not happen")

    monkeypatch.setattr(record_fixtures, "build_judge", fail_provider)

    with pytest.raises(RuntimeError, match="estimate"):
        asyncio.run(
            record_fixtures.run(
                output_path=tmp_path / "ci_subset_v2.json",
                max_cost_usd=0.01,
            )
        )

    assert called is False
