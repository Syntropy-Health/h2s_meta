"""Muse Glimmer chat-template output -> the member-facing reply (pure; no torch, no transformers).

The template renders an assistant reply as ``<|start|>assistant to=user<|message|>TEXT<|eot|>``
(reasoning goes to ``to=self``, tool calls to ``to=<tool>``), and its generation prompt
stops at ``<|start|>assistant``. So a generation starts with the plain-text recipient
header `` to=user``: decoding with ``skip_special_tokens=True`` leaks it into the reply
and keeps any reasoning. Decode with special tokens kept and pass the result here.
"""

from __future__ import annotations

import re

GENERATION_PROMPT = "<|start|>assistant"
_START = "<|start|>"
_MESSAGE = "<|message|>"
_CLOSE = re.compile(r"<\|(?:eot|eom|end_of_text)\|>")
_PAD = "<|finetune_right_pad|>"


def extract_reply(continuation: str) -> str:
    """The last ``to=user`` (or recipient-less) assistant message in a generation; ``""`` if none.

    ``continuation`` is the decoded new tokens after :data:`GENERATION_PROMPT`, special
    tokens kept. Reasoning (``to=self``), tool calls and unfinished headers are not replies.
    """
    text = (GENERATION_PROMPT + continuation).replace(_PAD, "")
    reply = ""
    for segment in text.split(_START)[1:]:
        header, sep, body = segment.partition(_MESSAGE)
        if not sep:
            continue
        words = header.split()
        if not words or words[0] != "assistant":
            continue
        recipients = [w[3:] for w in words[1:] if w.startswith("to=")]
        if recipients and recipients[-1] != "user":
            continue
        body = _CLOSE.split(body, maxsplit=1)[0]
        reply = body.strip()
    return reply


__all__ = ["GENERATION_PROMPT", "extract_reply"]
