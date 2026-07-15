"""Thin Qwen client with a per-run token ledger.

Every arm (single-agent, self-consistency, vanilla MAD, Majalis) calls through
here so token/latency accounting is identical and un-gameable by construction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

from .config import QWEN_BASE_URL, require

_RETRIES = 3
_BACKOFF_S = 2.0

# USD per 1M tokens (input, output) — Qwen Cloud base rates, <=256K tier
# (docs.qwencloud.com/developer-guides/getting-started/pricing, 2026-07).
# Re-verified 2026-07-12 against the Model Studio international (Singapore)
# price list and the real July bill: within 10% of list, conservative vs
# current promo discounts.
PRICES = {
    "qwen3.7-max": (2.50, 7.50),
    "qwen3.7-plus": (0.40, 1.60),
    "qwen3.6-flash": (0.25, 1.50),
}


def call_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


@dataclass
class Call:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class Ledger:
    calls: list[Call] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def wall_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(call_cost_usd(c.model, c.prompt_tokens, c.completion_tokens)
                   for c in self.calls)

    def as_dict(self) -> dict:
        return {
            "n_calls": len(self.calls),
            "total_tokens": self.total_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": round(self.wall_latency_s, 2),
            "cost_usd": round(self.cost_usd, 6),
        }


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=require("DASHSCOPE_API_KEY"), base_url=QWEN_BASE_URL)
    return _client


def chat(
    model: str,
    messages: list[dict],
    *,
    ledger: Ledger,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    seed: int | None = None,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            t0 = time.monotonic()
            resp = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            latency = time.monotonic() - t0
            usage = resp.usage
            ledger.calls.append(
                Call(model, usage.prompt_tokens, usage.completion_tokens, latency)
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — retry transient API errors uniformly
            last_exc = exc
            if "AllocationQuota" in str(exc) or "401" in str(exc):
                raise  # quota/auth won't heal by retrying
            time.sleep(_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"chat failed after {_RETRIES} retries: {last_exc}")
