import json

from evals.agreement import cohen_kappa, load_labels, paired


def test_perfect_agreement():
    assert cohen_kappa([("y", "y"), ("n", "n"), ("y", "y")]) == 1.0


def test_textbook_example():
    a = ["y", "y", "n", "n"]
    b = ["y", "n", "n", "n"]
    assert cohen_kappa(list(zip(a, b))) == 0.5


def test_disjoint_ids_ignored():
    assert paired({"1": "y", "2": "n"}, {"2": "n", "3": "y"}) == [("n", "n")]


def test_load_labels_skips_non_labels(tmp_path):
    p = tmp_path / "j.json"
    items = [
        {"id": "a", "correctness": "correct"},
        {"id": "b", "correctness": "refused"},
        {"id": "c", "correctness": "unparsed"},
        {"id": "d"},
    ]
    p.write_text(json.dumps({"items": items}), encoding="utf-8")
    assert load_labels(p, "correctness") == {"a": "correct"}
