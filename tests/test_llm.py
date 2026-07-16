"""Unit tests for llm.py — a fake OpenAI-shaped client, no live Qwen calls."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import majalis.llm as llm
from majalis.llm import Call, Ledger, call_cost_usd


def _resp(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens),
    )


class _FakeCompletions:
    """Pops one entry per call: an Exception is raised, anything else returned."""

    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _FakeClient:
    def __init__(self, side_effects):
        self.chat = SimpleNamespace(completions=_FakeCompletions(side_effects))


def _install_fake_client(monkeypatch, side_effects) -> _FakeClient:
    monkeypatch.setattr(llm, "_BACKOFF_S", 0.0)  # keep retry tests instant
    fake = _FakeClient(side_effects)
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    return fake


# --- Ledger cost arithmetic ------------------------------------------------------

def test_ledger_cost_arithmetic():
    ledger = Ledger()
    ledger.calls.append(Call("qwen3.7-max", 1000, 500, 1.5))
    ledger.calls.append(Call("qwen3.7-plus", 2000, 1000, 2.0))
    assert ledger.total_tokens == 1000 + 500 + 2000 + 1000
    assert ledger.completion_tokens == 500 + 1000
    assert ledger.wall_latency_s == pytest.approx(3.5)
    expected_cost = (call_cost_usd("qwen3.7-max", 1000, 500)
                     + call_cost_usd("qwen3.7-plus", 2000, 1000))
    assert ledger.cost_usd == pytest.approx(expected_cost)
    d = ledger.as_dict()
    assert d["n_calls"] == 2
    assert d["total_tokens"] == 4500
    assert d["cost_usd"] == pytest.approx(expected_cost, abs=1e-6)


def test_ledger_unknown_model_prices_at_zero():
    ledger = Ledger()
    ledger.calls.append(Call("some-unpriced-model", 1000, 1000, 1.0))
    assert ledger.cost_usd == 0.0


# --- chat(): retry-then-succeed --------------------------------------------------

def test_chat_retries_transient_errors_then_succeeds(monkeypatch):
    fake = _install_fake_client(monkeypatch, [
        Exception("transient 500"), Exception("transient 500"), _resp("final answer"),
    ])
    ledger = Ledger()
    out = llm.chat("qwen3.7-max", [{"role": "user", "content": "hi"}], ledger=ledger)
    assert out == "final answer"
    assert fake.chat.completions.calls == 3
    assert len(ledger.calls) == 1  # only the successful call is recorded
    assert ledger.total_tokens == 15


# --- chat(): retry exhaustion -> RuntimeError -------------------------------------

def test_chat_retry_exhaustion_raises_runtime_error(monkeypatch):
    fake = _install_fake_client(monkeypatch, [
        Exception("boom 1"), Exception("boom 2"), Exception("boom 3"),
    ])
    ledger = Ledger()
    with pytest.raises(RuntimeError, match="chat failed after 3 retries"):
        llm.chat("qwen3.7-max", [{"role": "user", "content": "hi"}], ledger=ledger)
    assert fake.chat.completions.calls == 3
    assert ledger.calls == []  # nothing succeeded, ledger untouched


# --- chat(): quota/auth fast-fail (no retry) --------------------------------------

@pytest.mark.parametrize("message", ["AllocationQuota exceeded", "401 Unauthorized"])
def test_chat_quota_and_auth_errors_fail_fast_without_retry(monkeypatch, message):
    fake = _install_fake_client(monkeypatch, [Exception(message)])
    ledger = Ledger()
    with pytest.raises(Exception, match=message.split()[0]):
        llm.chat("qwen3.7-max", [{"role": "user", "content": "hi"}], ledger=ledger)
    assert fake.chat.completions.calls == 1  # not retried
