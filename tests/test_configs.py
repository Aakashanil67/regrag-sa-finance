import os
import subprocess
import sys

from evals import configs


def test_load_has_every_named_config():
    assert set(configs.load()) == {"default", "baseline", "wordpiece", "bge", "hybrid"}


def test_non_default_configs_use_their_own_store():
    for name, overrides in configs.load().items():
        if name != "default":
            assert overrides["REGRAG_CHROMA_DIR"].startswith("chroma_variants/")


def test_default_leaves_settings_at_defaults():
    env = {k: v for k, v in os.environ.items() if not k.startswith("REGRAG_")}
    out = subprocess.run(
        [sys.executable, "-m", "evals.configs", "settings", "default"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    assert '"chunk_mode": "wordpiece"' in out and '"chunk_target": 240' in out
    assert '"chroma_dir": "chroma"' in out
