import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _example_questions() -> list[str]:
    tree = ast.parse((ROOT / "app" / "chat.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "EXAMPLE_QUESTIONS":
            return ast.literal_eval(node.value)
    raise AssertionError("EXAMPLE_QUESTIONS not found")


def test_examples_are_five_unique_strings():
    questions = _example_questions()
    assert len(questions) == 5
    assert len(set(questions)) == 5
    assert all(isinstance(q, str) and q.strip() for q in questions)


def test_examples_were_served_in_the_final_test_run():
    run = json.loads((ROOT / "reports" / "runs" / "test-final.json").read_text(encoding="utf-8"))
    served = {item["question"] for item in run["items"] if not item["refused"]}
    for question in _example_questions():
        assert question in served
