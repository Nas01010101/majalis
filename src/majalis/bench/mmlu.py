"""MMLU (multiple-choice) head-to-head: single-agent vs the Majalis gate
adaptation vs vanilla multi-agent debate (MAD).

Unlike the synthetic belief-board streams (where every arm saturates near 100%
and accuracy cannot separate the society from a single agent), MMLU's
reasoning-heavy subjects are a regime where multi-agent debate is known to lift
accuracy (Du et al. 2023; iMAD; SELENE). This benchmark exists specifically to
test the society/gate on a task that is NOT saturated, so debate has room to
help and the gate can be judged on whether it captures most of that benefit at a
fraction of the cost.

As with GSM8K, there is no evidence stream and no belief board here, so the
learned world model (wm.py) is NOT used: its trained features (wrong_now,
superseded_next) do not exist for a bare multiple-choice question. The
"majalis-gated" arm wires the same propose -> gate -> skeptic -> judge shape onto
the only signals available for a bare question: the proposer's verbalized
confidence and (only when that is ambiguous) a cheap k=2 self-consistency
sample. This is an UNCALIBRATED gate; no conformal guarantee is claimed here.

    python scripts/mmlu_bench.py --arms single,majalis-gated,mad --n 150
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..config import MODEL_FAST, MODEL_MID, MODEL_STRONG
from ..handoffs import parse_json_block
from ..llm import Ledger, chat

_LETTERS = ["A", "B", "C", "D"]
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))


def extract_pred(model_output: str) -> str | None:
    """Last well-formed {"answer": "<letter>"} JSON object wins; falls back to
    the last bare A-D letter in the text (mirrors gsm8k.extract_pred)."""
    braces = re.findall(r"\{[^{}]*\}", model_output, re.DOTALL)
    for cand in reversed(braces):
        parsed = parse_json_block(cand)
        if isinstance(parsed, dict) and "answer" in parsed:
            m = _LETTER_RE.search(str(parsed["answer"]).strip().upper())
            if m:
                return m.group(1)
    letters = _LETTER_RE.findall(model_output.upper())
    return letters[-1] if letters else None


def letter_match(pred: str | None, gold: str) -> bool:
    return pred is not None and pred == gold


# --- Prompts --------------------------------------------------------------------

_PROPOSE_RULES = (
    "Reason step by step, then on the LAST line output ONLY this JSON object "
    '(no other text on that line): {"answer": "<one letter A, B, C, or D>", '
    '"confidence": <float 0..1, your honest probability this letter is correct>}'
)
_SC_RULES = "Give ONLY the single letter (A, B, C, or D) of the best answer, nothing else."


@dataclass
class GateFeatures:
    confidence: float
    disagreement: float
    sampled: bool


@dataclass
class GateDecision:
    fire: bool
    reason: str
    features: GateFeatures

    def as_dict(self) -> dict:
        return {"fired": self.fire, "reason": self.reason,
                "confidence": round(self.features.confidence, 3),
                "disagreement": round(self.features.disagreement, 3),
                "sampled": self.features.sampled}


@dataclass
class MmluResult:
    answer: str | None
    ledger: Ledger
    gate: dict | None  # None for arms without a gate (single, mad)


# Uncalibrated thresholds (no held-out calibration set for MMLU in scope), stated
# honestly rather than borrowing the session bench's conformal alpha.
GATE_CONF_SKIP = 0.85
GATE_FIRE_CONF = 0.6
_K_SAMPLES = 2


def _qtext(question: str, choices: list[str]) -> str:
    return f"Question:\n{question}\n\nChoices:\n{format_choices(choices)}"


def _propose(qtext: str, ledger: Ledger, model: str, seed: int) -> tuple[str | None, float, str]:
    out = chat(model, [{"role": "user", "content": f"{qtext}\n\n{_PROPOSE_RULES}"}],
               ledger=ledger, temperature=0.0, max_tokens=768, seed=seed)
    confidence = 0.5
    for cand in reversed(re.findall(r"\{[^{}]*\}", out, re.DOTALL)):
        parsed = parse_json_block(cand)
        if isinstance(parsed, dict) and "confidence" in parsed:
            try:
                confidence = max(0.0, min(1.0, float(parsed["confidence"])))
            except (TypeError, ValueError):
                pass
            break
    return extract_pred(out), confidence, out


def _sample_disagreement(qtext: str, proposal: str | None,
                         ledger: Ledger, model: str, seed: int) -> float:
    """k=2 short high-temperature samples; disagreement = 1 - modal letter frequency."""
    votes: list[str | None] = [proposal] if proposal else []
    for i in range(_K_SAMPLES):
        out = chat(model, [{"role": "user", "content": f"{qtext}\n\n{_SC_RULES}"}],
                   ledger=ledger, temperature=1.0, max_tokens=8, seed=seed * 100 + i)
        votes.append(extract_pred(out))
    votes = [v for v in votes if v]
    if not votes:
        return 1.0
    modal = Counter(votes).most_common(1)[0][1]
    return 1.0 - modal / len(votes)


def _gate_decide(qtext: str, proposal: str | None, confidence: float,
                 ledger: Ledger, seed: int) -> GateDecision:
    if confidence >= GATE_CONF_SKIP:
        return GateDecision(False, "confidence-accept",
                            GateFeatures(confidence, 0.0, False))
    disagreement = _sample_disagreement(qtext, proposal, ledger, MODEL_FAST, seed)
    feats = GateFeatures(confidence, disagreement, True)
    if confidence < GATE_FIRE_CONF:
        return GateDecision(True, "low-confidence", feats)
    if disagreement > 0.0:
        return GateDecision(True, "self-consistency-disagreement", feats)
    return GateDecision(False, "self-consistency-agrees", feats)


def _skeptic(qtext: str, rationale: str, proposal: str | None,
             ledger: Ledger, model: str, seed: int) -> str:
    prompt = (
        "You are an adversarial checker for a multiple-choice question. The proposed "
        f"answer may be wrong.\n\n{qtext}\n\nProposed reasoning:\n{rationale}\n\n"
        f"Proposed answer: {proposal}\n\nFind the specific flaw (if any) in the reasoning "
        "or in eliminating the other options. If you find no error, say so explicitly."
    )
    return chat(model, [{"role": "user", "content": prompt}],
                ledger=ledger, temperature=0.7, max_tokens=512, seed=seed)


def _judge(qtext: str, rationale: str, proposal: str | None,
           attack: str, ledger: Ledger, model: str, seed: int) -> str | None:
    prompt = (
        f"{qtext}\n\nA proposer chose {proposal} with this reasoning:\n{rationale}\n\n"
        f"A skeptic objected:\n{attack}\n\nRe-derive the answer yourself from scratch, "
        f"then decide the correct letter. {_PROPOSE_RULES}"
    )
    out = chat(model, [{"role": "user", "content": prompt}],
               ledger=ledger, temperature=0.0, max_tokens=768, seed=seed)
    return extract_pred(out)


def single_arm(question: str, choices: list[str], *, model: str = MODEL_STRONG,
               seed: int = 0) -> MmluResult:
    ledger = Ledger()
    pred, _c, _o = _propose(_qtext(question, choices), ledger, model, seed)
    return MmluResult(answer=pred, ledger=ledger, gate=None)


def majalis_gated_arm(question: str, choices: list[str], *, seed: int = 0) -> MmluResult:
    ledger = Ledger()
    qtext = _qtext(question, choices)
    pred, confidence, out = _propose(qtext, ledger, MODEL_STRONG, seed)
    decision = _gate_decide(qtext, pred, confidence, ledger, seed)
    if not decision.fire:
        return MmluResult(answer=pred, ledger=ledger, gate=decision.as_dict())
    attack = _skeptic(qtext, out, pred, ledger, MODEL_MID, seed)
    corrected = _judge(qtext, out, pred, attack, ledger, MODEL_STRONG, seed)
    final = corrected if corrected is not None else pred
    return MmluResult(answer=final, ledger=ledger, gate=decision.as_dict())


def mad_arm(question: str, choices: list[str], *, seed: int = 0,
            n_agents: int = 3, rounds: int = 2) -> MmluResult:
    """Vanilla multi-agent debate baseline (Du et al. 2023 shape): n agents
    propose independently, then revise for `rounds-1` rounds while seeing each
    other's answers; final = majority vote. Homogeneous (same strong model),
    which is the standard MAD baseline the sparse-debate literature compares to."""
    ledger = Ledger()
    qtext = _qtext(question, choices)
    answers: list[str | None] = []
    rationales: list[str] = []
    for a in range(n_agents):
        pred, _c, out = _propose(qtext, ledger, MODEL_STRONG, seed * 10 + a)
        answers.append(pred)
        rationales.append(out)
    for r in range(rounds - 1):
        new_answers: list[str | None] = []
        new_rationales: list[str] = []
        for a in range(n_agents):
            others = "\n\n".join(
                f"Agent {i + 1} answered {answers[i]}: {rationales[i][:400]}"
                for i in range(n_agents) if i != a)
            prompt = (f"{qtext}\n\nOther agents' answers and reasoning:\n{others}\n\n"
                      f"Reconsider and give your updated answer. {_PROPOSE_RULES}")
            out = chat(MODEL_STRONG, [{"role": "user", "content": prompt}],
                       ledger=ledger, temperature=0.0, max_tokens=768,
                       seed=seed * 100 + r * 10 + a)
            new_answers.append(extract_pred(out))
            new_rationales.append(out)
        answers, rationales = new_answers, new_rationales
    votes = [x for x in answers if x]
    final = Counter(votes).most_common(1)[0][0] if votes else None
    return MmluResult(answer=final, ledger=ledger, gate=None)


ARMS_MMLU = {
    "single": single_arm,
    "majalis-gated": majalis_gated_arm,
    "mad": mad_arm,
}
