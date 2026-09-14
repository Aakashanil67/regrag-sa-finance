"""Check current release evidence without calling a model or provider.

Usage: ``python -m scripts.check_release``
"""

import argparse
import sys
from pathlib import Path

from evals.evidence import REGISTRY_PATH, current_evidence_errors, load_registry
from evals.sealed_run import ProtocolValidationError, validate_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--protocol", type=Path, help="provider-free schema-2 protocol preflight")
    args = parser.parse_args(argv)

    try:
        if args.protocol:
            validate_protocol(args.protocol)
        registry = load_registry(args.registry)
        errors = current_evidence_errors(registry)
    except (OSError, ValueError, ProtocolValidationError) as exc:
        print(f"NOT RELEASE-READY: {exc}")
        return 1

    if errors:
        print("NOT RELEASE-READY")
        for error in errors:
            print(f"- {error}")
        return 1

    print("RELEASE-READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
