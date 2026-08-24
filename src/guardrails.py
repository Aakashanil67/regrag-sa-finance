"""Prompt-injection detection: observability, not a block.

This is deliberately a detector, not a gate — flagging a question doesn't stop it from being
answered. The domain is narrow (SA financial regulation Q&A), so a legitimate question is
extremely unlikely to trip these patterns, but "extremely unlikely" isn't zero, and refusing a
real user's question because it contains the word "ignore" is a worse failure than letting a
flagged-but-harmless one through. The actual defense against an injected instruction doing
anything is the system prompt in rag.py (answer only from context, never adopt a different
persona) — this module exists so an attempt shows up in the query log for review, which a silent
system prompt defense alone doesn't give you.
"""

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
