"""Smoke test: one tiny call per backbone. Exits non-zero if Qwen quota/auth is broken."""
from __future__ import annotations

import sys

from openai import OpenAI

sys.path.insert(0, "src")
from agora.config import MODEL_FAST, MODEL_MID, MODEL_STRONG, QWEN_BASE_URL, require


def main() -> int:
    client = OpenAI(api_key=require("DASHSCOPE_API_KEY"), base_url=QWEN_BASE_URL)
    failures = 0
    for model in (MODEL_FAST, MODEL_MID, MODEL_STRONG):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
                max_tokens=8,
            )
            print(f"{model}: {resp.choices[0].message.content!r}")
        except Exception as exc:  # noqa: BLE001 — report every backbone, then fail
            failures += 1
            print(f"{model}: FAILED — {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
