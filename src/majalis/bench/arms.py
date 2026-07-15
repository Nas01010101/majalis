"""Benchmark arms. Every arm gets the same task text and the same ledger
accounting; the Majalis arm is wired in once the society core lands.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..config import MODEL_STRONG
from ..llm import Ledger, chat
from .tasks import Task

_ANSWER_RULES = (
    "Think step by step, then give your final answer on the last line as: "
    "ANSWER: <short answer>"
)


def _extract(answer: str) -> str:
    for line in reversed(answer.strip().splitlines()):
        if line.upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return answer.strip().splitlines()[-1] if answer.strip() else ""


@dataclass
class ArmResult:
    answer: str
    ledger: Ledger
    transcript: list[dict]


def _user_msg(task: Task) -> str:
    return f"Evidence:\n{task.context}\n\n{task.question}\n\n{_ANSWER_RULES}"


def single_cot(task: Task, *, model: str = MODEL_STRONG, seed: int = 0) -> ArmResult:
    ledger = Ledger()
    out = chat(model, [{"role": "user", "content": _user_msg(task)}],
               ledger=ledger, temperature=0.0, seed=seed)
    return ArmResult(_extract(out), ledger, [{"role": "single", "text": out}])


def self_consistency(task: Task, *, k: int = 5, model: str = MODEL_STRONG,
                     seed: int = 0) -> ArmResult:
    ledger = Ledger()
    votes: list[str] = []
    transcript = []
    for i in range(k):
        out = chat(model, [{"role": "user", "content": _user_msg(task)}],
                   ledger=ledger, temperature=0.8, seed=seed * 1000 + i)
        votes.append(_extract(out).lower())
        transcript.append({"role": f"sample-{i}", "text": out})
    majority = Counter(votes).most_common(1)[0][0]
    return ArmResult(majority, ledger, transcript)


def vanilla_mad(task: Task, *, n_agents: int = 3, n_rounds: int = 3,
                model: str = MODEL_STRONG, seed: int = 0) -> ArmResult:
    """Fixed-round, all-talk multi-agent debate — the hundred-submissions baseline."""
    ledger = Ledger()
    transcript: list[dict] = []
    positions = [""] * n_agents
    for rnd in range(n_rounds):
        for i in range(n_agents):
            others = "\n\n".join(
                f"Agent {j + 1} said:\n{positions[j]}"
                for j in range(n_agents) if j != i and positions[j]
            )
            prompt = _user_msg(task)
            if others:
                prompt += (
                    f"\n\nOther agents' current answers:\n{others}\n\n"
                    "Consider their reasoning, then give your (possibly updated) answer."
                )
            out = chat(model, [{"role": "user", "content": prompt}],
                       ledger=ledger, temperature=0.7, seed=seed * 100 + rnd * 10 + i)
            positions[i] = out
            transcript.append({"role": f"agent{i + 1}-r{rnd + 1}", "text": out})
    votes = [_extract(p).lower() for p in positions]
    majority = Counter(votes).most_common(1)[0][0]
    return ArmResult(majority, ledger, transcript)


ARMS = {
    "single": single_cot,
    "sc5": self_consistency,
    "mad": vanilla_mad,
    # "majalis" registered by majalis.society once it exists
}
