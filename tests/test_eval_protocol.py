import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_hash_matches_the_sealed_test_set():
    protocol = json.loads((ROOT / "evals" / "protocol_test.json").read_text(encoding="utf-8"))
    data = (ROOT / protocol["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == protocol["sha256"]
