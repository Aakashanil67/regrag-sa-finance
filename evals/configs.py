"""Named pipeline configs: sets of REGRAG_* environment overrides read by src.config at import.

python -m evals.configs show wordpiece      # effective settings under that config
python -m evals.configs settings wordpiece  # same, one JSON line (used by checks)
python -m evals.configs build wordpiece     # build that config's vector store
"""

import json
import os
import sys
from pathlib import Path

CONFIGS_PATH = Path(__file__).with_name("configs.json")


def load() -> dict[str, dict[str, str]]:
    return json.loads(CONFIGS_PATH.read_text(encoding="utf-8"))


def apply(name: str) -> dict[str, str]:
    """Set the environment for `name`. Must run before anything imports src.config."""
    overrides = load()[name]
    if overrides and "src.config" in sys.modules:
        raise RuntimeError("apply() ran after src.config was imported; settings would be stale")
    for key in [k for k in os.environ if k.startswith("REGRAG_")]:
        del os.environ[key]
    os.environ.update(overrides)
    return overrides


def main(argv: list[str] | None = None) -> None:
    cmd, name = (argv or sys.argv[1:])[:2]
    apply(name)
    import src.config as config

    if cmd in ("show", "settings"):
        print(json.dumps(config.effective_settings(), sort_keys=True))
    elif cmd == "build":
        from src.store import rebuild

        print(json.dumps(rebuild(), default=str))
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
