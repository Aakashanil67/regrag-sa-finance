"""Answer the example questions once at image build so visitors get them from the cache."""

import ast
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.cache import set_cached  # noqa: E402
from src.rag import answer_question  # noqa: E402

K = 5  # the API's default, see src.obslog.timed_answer


def example_questions() -> list[str]:
    # parsed, not imported: app/chat.py pulls in Streamlit and runs the page on import
    tree = ast.parse((ROOT / "app" / "chat.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "EXAMPLE_QUESTIONS":
            return ast.literal_eval(node.value)
    raise SystemExit("EXAMPLE_QUESTIONS not found in app/chat.py")


def main() -> None:
    for question in example_questions():
        start = time.perf_counter()
        result = answer_question(question, k=K)
        set_cached(question, result, k=K)
        print(f"{time.perf_counter() - start:6.1f}s  {question}", flush=True)


if __name__ == "__main__":
    main()
