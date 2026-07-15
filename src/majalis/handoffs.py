"""Typed artifacts passed between roles. Never free chat between agents —
structured handoffs dodge MAST's inter-agent-misalignment failure cluster
(arXiv:2503.13657) and beat dialogue on quality (MetaGPT, arXiv:2308.00352).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ExtractedFact:
    entity: str
    attribute: str
    value: str
    date: str  # as it appears in evidence, e.g. "Mar 2025"


@dataclass
class Proposal:
    answer: str
    rationale: str
    support_keys: list[str]  # belief-board keys the answer rests on
    confidence: float  # author's self-report, 0..1 (advisory only)
    author: str = ""


@dataclass
class Challenge:
    target_key: str
    attack: str  # the skeptic's specific objection
    sub_questions: list[str]  # binary checkable decomposition
    author: str = ""


@dataclass
class Verdict:
    target_key: str
    upheld: bool  # True = proposer's belief survives the challenge
    corrected_value: str | None  # set when the challenge flips the belief
    rationale: str
    author: str = ""


@dataclass
class DebateTrace:
    """Full decision trace for one task — feeds the dashboard and audits."""
    task_id: str
    gate: dict = field(default_factory=dict)  # why debate fired / was skipped
    events: list[dict] = field(default_factory=list)

    def log(self, kind: str, **payload) -> None:
        self.events.append({"kind": kind, **payload})


def parse_json_block(text: str) -> dict | list | None:
    """Lenient JSON extraction from an LLM reply (bare, fenced, or embedded)."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    brace = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for cand in candidates:
        try:
            return json.loads(cand.strip())
        except (json.JSONDecodeError, ValueError):
            continue
    return None
