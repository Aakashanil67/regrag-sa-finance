"""Flag possible prompt injection in user questions for the usage log.

Pattern matches do not block a request. Domain questions can contain the same phrases, so the
system prompt and citation validator still decide whether an answer can be served."""

import re

_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"ignore (all |any |the )?(previous|prior|above) instructions",
        r"disregard (all |any |the )?(previous|prior|above) (instructions|rules|prompt)",
        r"you are now\b",
        r"act as (a|an)\b",
        r"reveal (your|the) (system )?prompt",
        r"repeat (your|the) (system )?prompt",
        r"new instructions\s*:",
        r"</?(system|instructions?)>",
    ]
]


def contains_injection_attempt(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
