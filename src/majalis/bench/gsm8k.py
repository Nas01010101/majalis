"""GSM8K head-to-head: single-agent vs an honest ADAPTATION of the Majalis
gate to single-turn math QA.

This is a different regime from the session/churn benches: there is no
evidence stream, no belief board, no supersession or rumor-vs-filing
distinction. The learned world model (wm.py) was trained on belief-board
features (wrong_now, superseded_next) that simply do not exist here, so it
is NOT used for this benchmark. Instead the "majalis-gated" arm below wires
the SAME trigger/debate shape (propose -> gate -> skeptic -> judge) onto the
only uncertainty signals available for a bare question: the proposer's own
verbalized confidence, and (only when that confidence is ambiguous) a cheap
k=2 self-consistency sample. This is explicitly a heuristic, UNCALIBRATED
gate (no conformal guarantee is claimed for this benchmark) — see the
GATE_* constants and `_gate_decide` docstring for exactly what it sees.

    python scripts/gsm8k_bench.py --arms single,majalis-gated --n 200
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import MODEL_FAST, MODEL_MID, MODEL_STRONG
from ..handoffs import parse_json_block
from ..llm import Ledger, chat

# --- Answer extraction --------------------------------------------------------

_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_gold(raw_answer: str) -> float:
    """GSM8K's canonical gold format: reasoning, then '#### <number>'."""
    m = _GOLD_RE.search(raw_answer)
    if not m:
        raise ValueError(f"no '#### <number>' in gold answer: {raw_answer!r}")
    return float(m.group(1).replace(",", ""))


def _to_float(token) -> float | None:
    if token is None:
        return None
    s = str(token).replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        m = _NUM_RE.search(s)
        return float(m.group(0).replace(",", "")) if m else None


def extract_pred(model_output: str) -> float | None:
    """Last well-formed {"answer": ..., ...} JSON object wins; falls back to
    the last bare number in the text (mirrors bench/arms.py's ANSWER: line
    convention for models that ignore the JSON instruction)."""
    braces = re.findall(r"\{[^{}]*\}", model_output, re.DOTALL)
    for cand in reversed(braces):
        parsed = parse_json_block(cand)
        if isinstance(parsed, dict) and "answer" in parsed:
            val = _to_float(parsed["answer"])
            if val is not None:
                return val
    nums = _NUM_RE.findall(model_output)
    return _to_float(nums[-1]) if nums else None


def numeric_match(pred: float | None, gold: float, tol: float = 1e-4) -> bool:
    return pred is not None and abs(pred - gold) <= tol


# --- Prompts --------------------------------------------------------------------

_PROPOSE_RULES = (
    "Think step by step, then on the LAST line output ONLY this JSON object "
    '(no other text on that line): {"answer": <final number>, "confidence": '
    "<float 0..1, your honest probability this number is correct>}"
)

_SC_RULES = "Give ONLY the final numeric answer, nothing else — no words, no units."


@dataclass
class GateFeatures:
    """Everything the gate is allowed to look at for this benchmark —
    documented exhaustively so the adaptation's limits are auditable."""
    confidence: float  # verbalized, from the propose call
    disagreement: float  # 1 - modal frequency over k self-consistency samples (0.0 if skipped)
    sampled: bool  # whether the k=2 sampler ran at all


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
class Gsm8kResult:
    answer: float | None
    ledger: Ledger
    gate: dict | None  # None for the single arm


# Uncalibrated thresholds (no held-out calibration set for GSM8K in scope —
# stated honestly rather than borrowing the session bench's conformal alpha,
# which was fit on belief-board risk scores this task doesn't produce).
GATE_CONF_SKIP = 0.85     # confidence >= this: accept with zero extra calls
GATE_FIRE_CONF = 0.6      # confidence below this: debate regardless of sampler
_K_SAMPLES = 2


def _propose(question: str, ledger: Ledger, model: str, seed: int) -> tuple[float | None, float, str]:
    prompt = f"Problem:\n{question}\n\n{_PROPOSE_RULES}"
    out = chat(model, [{"role": "user", "content": prompt}],
               ledger=ledger, temperature=0.0, max_tokens=768, seed=seed)
    braces = re.findall(r"\{[^{}]*\}", out, re.DOTALL)
    confidence = 0.5
    for cand in reversed(braces):
        parsed = parse_json_block(cand)
        if isinstance(parsed, dict) and "confidence" in parsed:
            try:
                confidence = max(0.0, min(1.0, float(parsed["confidence"])))
            except (TypeError, ValueError):
                pass
            break
    return extract_pred(out), confidence, out


def _sample_disagreement(question: str, proposal_answer: float | None,
                         ledger: Ledger, model: str, seed: int) -> float:
    """CSD-style cheap tiebreaker: k=2 short samples at high temperature;
    disagreement = 1 - modal frequency, votes bucketed by numeric_match so
    '18' and '18.0' count as the same vote."""
    prompt = f"Problem:\n{question}\n\n{_SC_RULES}"
    votes: list[float | None] = []
    if proposal_answer is not None:
        votes.append(proposal_answer)
    for i in range(_K_SAMPLES):
        out = chat(model, [{"role": "user", "content": prompt}],
                   ledger=ledger, temperature=1.0, max_tokens=16,
                   seed=seed * 100 + i)
        votes.append(_to_float(out))
    if not votes:
        return 1.0
    buckets: list[float] = []
    counts: list[int] = []
    for v in votes:
        matched = False
        for i, b in enumerate(buckets):
            if v is not None and numeric_match(v, b):
                counts[i] += 1
                matched = True
                break
        if not matched:
            buckets.append(v if v is not None else float("nan"))
            counts.append(1)
    modal = max(counts)
    return 1.0 - modal / len(votes)


def _gate_decide(question: str, proposal_answer: float | None, confidence: float,
                 ledger: Ledger, seed: int) -> GateDecision:
    """The adapted trigger. Inputs are STRICTLY: verbalized confidence from
    the propose call, and (only when confidence is ambiguous) k=2 cheap
    self-consistency disagreement. No belief-board signal exists for a bare
    question, so weak-source/supersession/wm.wrong_now are all absent by
    construction — this is the honest floor of what a single-turn gate can
    see, not a reduced version of the full gate hiding extra state."""
    if confidence >= GATE_CONF_SKIP:
        feats = GateFeatures(confidence=confidence, disagreement=0.0, sampled=False)
        return GateDecision(fire=False, reason="confidence-accept", features=feats)

    disagreement = _sample_disagreement(question, proposal_answer, ledger, MODEL_FAST, seed)
    feats = GateFeatures(confidence=confidence, disagreement=disagreement, sampled=True)

    if confidence < GATE_FIRE_CONF:
        return GateDecision(fire=True, reason="low-confidence", features=feats)
    if disagreement > 0.0:
        return GateDecision(fire=True, reason="self-consistency-disagreement", features=feats)
    return GateDecision(fire=False, reason="self-consistency-agrees", features=feats)


def _skeptic(question: str, rationale: str, proposal_answer: float | None,
            ledger: Ledger, model: str, seed: int) -> str:
    prompt = (
        "You are an adversarial math checker. A proposed solution may contain "
        f"an arithmetic or reasoning error.\n\nProblem:\n{question}\n\n"
        f"Proposed solution:\n{rationale}\n\nProposed final answer: {proposal_answer}\n\n"
        "Find the specific step (if any) where the reasoning is wrong. Be concrete: "
        "quote the faulty step and explain the correct arithmetic. If you find no "
        "error, say so explicitly."
    )
    return chat(model, [{"role": "user", "content": prompt}],
               ledger=ledger, temperature=0.7, max_tokens=512, seed=seed)


def _judge(question: str, rationale: str, proposal_answer: float | None,
          attack: str, ledger: Ledger, model: str, seed: int) -> float | None:
    prompt = (
        f"Problem:\n{question}\n\nA proposer answered {proposal_answer} with this "
        f"reasoning:\n{rationale}\n\nA skeptic raised this objection:\n{attack}\n\n"
        "Re-derive the answer yourself from scratch, then decide: is the proposer's "
        f"answer correct? {_PROPOSE_RULES}"
    )
    out = chat(model, [{"role": "user", "content": prompt}],
              ledger=ledger, temperature=0.0, max_tokens=768, seed=seed)
    return extract_pred(out)


# --- BrowseConf baseline arm (arXiv:2510.23458) --------------------------------
# BrowseConf's core idea for web agents: after a proposed answer, ask the model
# to VERBALIZE a confidence score in a dedicated call and re-attempt (their
# paper: another full rollout) when it falls below a calibrated threshold —
# they report a strong confidence/accuracy correlation (near-zero accuracy
# below ~70% verbalized confidence, >2x the average above ~95%). We adapt the
# trigger (not the re-attempt mechanism, which doesn't exist for a one-shot
# math question) as a baseline GATE arm for GSM8K: the confidence signal comes
# from a SEPARATE cheap fast-model call (not free, unlike majalis-gated's
# confidence which rides along with the strong model's own answer) — this is
# the point of comparison: does a dedicated confidence call earn its cost
# over reading confidence off the answering call for free?
BROWSECONF_FIRE_CONF = 0.7  # BrowseConf's own reported accuracy cliff is ~0.70

_CONF_ONLY_PROMPT = (
    "Problem:\n{question}\n\nA model proposed this answer: {answer}\n\n"
    "Rate your confidence that this answer is correct, as a float 0..1. "
    'Output ONLY: {{"confidence": <float 0..1>}}'
)


def _confidence_only(question: str, answer: float | None, ledger: Ledger,
                     model: str, seed: int) -> float:
    prompt = _CONF_ONLY_PROMPT.format(question=question, answer=answer)
    out = chat(model, [{"role": "user", "content": prompt}],
              ledger=ledger, temperature=0.0, max_tokens=64, seed=seed)
    parsed = parse_json_block(out)
    if isinstance(parsed, dict) and "confidence" in parsed:
        try:
            return max(0.0, min(1.0, float(parsed["confidence"])))
        except (TypeError, ValueError):
            pass
    nums = re.findall(r"0?\.\d+|[01](?:\.0+)?", out)
    return max(0.0, min(1.0, float(nums[0]))) if nums else 0.5


def browseconf_arm(question: str, *, seed: int = 0) -> Gsm8kResult:
    ledger = Ledger()
    pred, _own_confidence, out = _propose(question, ledger, MODEL_STRONG, seed)
    confidence = _confidence_only(question, pred, ledger, MODEL_FAST, seed)
    if confidence >= BROWSECONF_FIRE_CONF:
        gate = {"fired": False, "reason": "confidence-accept", "confidence": round(confidence, 3)}
        return Gsm8kResult(answer=pred, ledger=ledger, gate=gate)

    attack = _skeptic(question, out, pred, ledger, MODEL_MID, seed)
    corrected = _judge(question, out, pred, attack, ledger, MODEL_STRONG, seed)
    final = corrected if corrected is not None else pred
    gate = {"fired": True, "reason": "low-confidence", "confidence": round(confidence, 3)}
    return Gsm8kResult(answer=final, ledger=ledger, gate=gate)


def single_arm(question: str, *, model: str = MODEL_STRONG, seed: int = 0) -> Gsm8kResult:
    ledger = Ledger()
    pred, _confidence, _out = _propose(question, ledger, model, seed)
    return Gsm8kResult(answer=pred, ledger=ledger, gate=None)


def majalis_gated_arm(question: str, *, seed: int = 0) -> Gsm8kResult:
    ledger = Ledger()
    pred, confidence, out = _propose(question, ledger, MODEL_STRONG, seed)
    decision = _gate_decide(question, pred, confidence, ledger, seed)
    if not decision.fire:
        return Gsm8kResult(answer=pred, ledger=ledger, gate=decision.as_dict())

    attack = _skeptic(question, out, pred, ledger, MODEL_MID, seed)
    corrected = _judge(question, out, pred, attack, ledger, MODEL_STRONG, seed)
    final = corrected if corrected is not None else pred
    return Gsm8kResult(answer=final, ledger=ledger, gate=decision.as_dict())


ARMS_GSM8K = {
    "single": single_arm,
    "majalis-gated": majalis_gated_arm,
    "browseconf": browseconf_arm,
}
