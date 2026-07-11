"""The Agora society: star-topology orchestrator over role agents.

The orchestrator is deterministic Python — the only component that touches
the belief board (sole-writer, mirrors Tenet's single-writer constraint).
Roles never talk to each other; they exchange typed artifacts through the
orchestrator (author≠validator enforced by construction: the Skeptic and
Judge never see the Proposer's identity, only its artifact).

v1 gate = doubt/confidence heuristic; the conformal control points
(CalibratedGate acceptance, CSD disagreement, EIG targeting, AnytimeAlarm)
land in wm.py and replace the heuristics behind the same seams.
"""
from __future__ import annotations

from .beliefs import BeliefBoard, parse_date_ord
from .bench.arms import ARMS, ArmResult
from .bench.tasks import Task
from .config import MODEL_FAST, MODEL_MID, MODEL_STRONG
from .handoffs import Challenge, DebateTrace, Proposal, Verdict, parse_json_block
from .llm import Ledger, chat
from .wm import AcceptGate, rank_targets

MAX_DEBATES_PER_TASK = 2

_GATE = AcceptGate()


# --- Roles (each is one prompt, one artifact) --------------------------------

def extract_facts(evidence: str, ledger: Ledger, model: str) -> list[dict]:
    prompt = (
        "Extract every dated factual assertion from the evidence as JSON.\n"
        f"Evidence:\n{evidence}\n\n"
        'Return a JSON list of objects: {"entity": str, "attribute": str, '
        '"value": str, "date": str, "source": str}. Use the date exactly as '
        'written (or "" if undated); source is the line\'s label, e.g. '
        '"Filing", "Blog recap", "Industry note", "Rumor". No commentary.'
    )
    out = chat(model, [{"role": "user", "content": prompt}],
               ledger=ledger, temperature=0.0, max_tokens=2048)
    parsed = parse_json_block(out)
    return parsed if isinstance(parsed, list) else []


def propose(task: Task, board: BeliefBoard, ledger: Ledger, model: str,
            correction_note: str = "") -> Proposal:
    prompt = (
        f"Question:\n{task.question}\n\n"
        f"Current belief state (most recent value per fact, extracted from "
        f"the dated evidence):\n{board.summary()}\n\n"
        f"{correction_note}"
        'Answer the question using the CURRENT beliefs. Return JSON: '
        '{"answer": str (short), "rationale": str, "support_keys": [str], '
        '"confidence": float 0..1}. support_keys must be keys from the '
        "belief state your answer rests on."
    )
    out = chat(model, [{"role": "user", "content": prompt}],
               ledger=ledger, temperature=0.0, max_tokens=1024)
    parsed = parse_json_block(out) or {}
    return Proposal(
        answer=str(parsed.get("answer", "")).strip(),
        rationale=str(parsed.get("rationale", "")),
        support_keys=[str(k).lower() for k in parsed.get("support_keys", [])],
        confidence=float(parsed.get("confidence", 0.5) or 0.5),
        author="proposer",
    )


def skeptic_challenge(task: Task, board: BeliefBoard, key: str,
                      proposal: Proposal, ledger: Ledger, model: str) -> Challenge:
    cur = board.current(key)
    prompt = (
        "You are an adversarial fact-checker. A proposed answer rests on a "
        "belief you must attack.\n"
        f"Belief under challenge: {key} = {cur.value if cur else '?'}\n"
        f"Recorded assertions for this key (the docket):\n{board.docket(key)}\n\n"
        f"Proposed answer: {proposal.answer}\nRationale: {proposal.rationale}\n\n"
        "Attack the belief: is the value stale, contradicted, sourced from an "
        "unreliable mention (filings are authoritative; blog/industry/rumor "
        "lines are not), or misread? "
        'Return JSON: {"attack": str, "sub_questions": [str]} '
        "where sub_questions are 2-4 binary-checkable questions whose answers "
        "settle the challenge (each answerable by pointing at a dated line)."
    )
    out = chat(model, [{"role": "user", "content": prompt}],
               ledger=ledger, temperature=0.7, max_tokens=1024)
    parsed = parse_json_block(out) or {}
    return Challenge(
        target_key=key,
        attack=str(parsed.get("attack", out[:300])),
        sub_questions=[str(q) for q in parsed.get("sub_questions", [])][:4],
        author="skeptic",
    )


def adjudicate(task: Task, board: BeliefBoard, proposal: Proposal,
               challenge: Challenge, ledger: Ledger, model: str) -> Verdict:
    cur = board.current(challenge.target_key)
    subq = "\n".join(f"- {q}" for q in challenge.sub_questions)
    prompt = (
        "You are a judge. Settle a challenge by answering each sub-question "
        "ONLY from the dated docket entries below (cite the deciding entry). "
        "Policy: filings are authoritative and the most recent FILING wins; "
        "blog recaps, industry notes and rumors never override a filing.\n"
        f"Docket for {challenge.target_key}:\n{board.docket(challenge.target_key)}\n\n"
        f"Belief: {challenge.target_key} = {cur.value if cur else '?'}\n"
        f"Attack: {challenge.attack}\nSub-questions:\n{subq}\n\n"
        'Return JSON: {"upheld": bool (true if the belief survives), '
        '"corrected_value": str|null (the correct current value if it does '
        'not), "rationale": str (cite the deciding dated lines)}.'
    )
    out = chat(model, [{"role": "user", "content": prompt}],
               ledger=ledger, temperature=0.0, max_tokens=1024)
    parsed = parse_json_block(out) or {}
    corrected = parsed.get("corrected_value")
    return Verdict(
        target_key=challenge.target_key,
        upheld=bool(parsed.get("upheld", True)),
        corrected_value=str(corrected).strip().lower() if corrected else None,
        rationale=str(parsed.get("rationale", "")),
        author="judge",
    )


# --- Orchestrator -------------------------------------------------------------

def run_agora(task: Task, *, seed: int = 0,
              model_strong: str = MODEL_STRONG,
              model_fast: str = MODEL_FAST,
              gate_mode: str = "wm",  # wm | always | never (ablations)
              eig_targeting: bool = True) -> ArmResult:
    ledger = Ledger()
    trace = DebateTrace(task_id=task.task_id)
    board = BeliefBoard()

    # 1. Perception: evidence -> keyed beliefs (supersession by date).
    for fact in extract_facts(task.context, ledger, model_fast):
        try:
            key = BeliefBoard.make_key(str(fact["entity"]), str(fact["attribute"]))
            outcome = board.assert_fact(key, str(fact["value"]),
                                        parse_date_ord(str(fact.get("date", ""))),
                                        source=str(fact.get("source", "")))
            trace.log("assert", key=key, value=str(fact["value"]), outcome=outcome)
        except (KeyError, TypeError):
            continue

    # 2. Proposal.
    proposal = propose(task, board, ledger, model_strong)
    trace.log("proposal", answer=proposal.answer,
              support=proposal.support_keys, confidence=proposal.confidence)

    # 3. Gate: is a debate worth the tokens? Calibrated accept decision
    #    (E[error | accepted] <= alpha once gate_calibration.json exists).
    decision = _GATE.decide(task, board, proposal, ledger, model_fast, seed=seed)
    if gate_mode == "always":
        decision.fire, decision.reason = True, "ablation:always-debate"
    elif gate_mode == "never":
        decision.fire, decision.reason = False, "ablation:never-debate"
    trace.gate = decision.as_dict()

    # 4. Debate the most informative doubted beliefs, EIG-ordered, bounded.
    if eig_targeting:
        targets = rank_targets(board, proposal.support_keys,
                               max_targets=MAX_DEBATES_PER_TASK, wm=_GATE.wm)
    else:  # ablation: naive order, no information-gain ranking
        targets = [k for k in proposal.support_keys
                   if board.current(k) is not None][:MAX_DEBATES_PER_TASK]
    if decision.fire and targets:
        adjudications = []
        for key in targets:
            # Skeptic runs on a DIFFERENT backbone than the proposer/judge:
            # model heterogeneity is the one debate lever with robust evidence
            # (arXiv:2502.08788 calls it the "universal antidote").
            challenge = skeptic_challenge(task, board, key, proposal, ledger, MODEL_MID)
            trace.log("challenge", key=key, attack=challenge.attack[:200])
            verdict = adjudicate(task, board, proposal, challenge, ledger, model_strong)
            trace.log("verdict", key=key, upheld=verdict.upheld,
                      corrected=verdict.corrected_value)
            if not verdict.upheld and verdict.corrected_value:
                # Write-back: adjudicated correction supersedes (moderator seam).
                board.assert_fact(key, verdict.corrected_value,
                                  board._now_ord + 1, source="debate")
                adjudications.append(f"{key}: CORRECTED to {verdict.corrected_value}")
            else:
                adjudications.append(f"{key}: belief UPHELD as stated")
        # Always re-propose after a debate: the original proposal may
        # contradict a belief the judge just upheld (observed failure mode).
        note = ("Adjudication results for the challenged beliefs:\n"
                + "\n".join(f"- {a}" for a in adjudications)
                + "\nAnswer strictly from the belief state below.\n\n")
        proposal = propose(task, board, ledger, model_strong, correction_note=note)
        trace.log("reproposal", answer=proposal.answer)

    result = ArmResult(proposal.answer, ledger,
                       [{"role": "trace", "text": ""}])
    result.transcript = [{"role": "trace", "gate": trace.gate,
                          "events": trace.events}]
    return result


class AgoraSession:
    """Incremental society over an evidence STREAM: ingest each batch once,
    answer every question from the compact board; debates read per-key
    dockets. This is where the world model's cost structure pays —
    O(stream) perception happens once, questions cost O(board)."""

    def __init__(self, *, seed: int = 0,
                 model_strong: str = MODEL_STRONG,
                 model_fast: str = MODEL_FAST,
                 gate_mode: str = "wm"):  # wm | always | never
        self.board = BeliefBoard()
        self.seed = seed
        self.model_strong = model_strong
        self.model_fast = model_fast
        self.gate_mode = gate_mode
        self.ingest_ledger = Ledger()  # perception cost, amortized over questions

    def ingest(self, lines: list[str]) -> None:
        for fact in extract_facts("\n".join(lines), self.ingest_ledger, self.model_fast):
            try:
                self.board.assert_fact(
                    BeliefBoard.make_key(str(fact["entity"]), str(fact["attribute"])),
                    str(fact["value"]),
                    parse_date_ord(str(fact.get("date", ""))),
                    source=str(fact.get("source", "")))
            except (KeyError, TypeError):
                continue

    def ask(self, task: Task) -> ArmResult:
        ledger = Ledger()
        trace = DebateTrace(task_id=task.task_id)
        board = self.board
        proposal = propose(task, board, ledger, self.model_strong)
        trace.log("proposal", answer=proposal.answer,
                  support=proposal.support_keys, confidence=proposal.confidence)
        decision = _GATE.decide(task, board, proposal, ledger, self.model_fast,
                                seed=self.seed)
        if self.gate_mode == "always":
            decision.fire, decision.reason = True, "ablation:always-debate"
        elif self.gate_mode == "never":
            decision.fire, decision.reason = False, "ablation:never-debate"
        trace.gate = decision.as_dict()
        targets = rank_targets(board, proposal.support_keys,
                               max_targets=MAX_DEBATES_PER_TASK, wm=_GATE.wm)
        if decision.fire and targets:
            adjudications = []
            for key in targets:
                challenge = skeptic_challenge(task, board, key, proposal,
                                              ledger, MODEL_MID)
                trace.log("challenge", key=key, attack=challenge.attack[:200])
                verdict = adjudicate(task, board, proposal, challenge,
                                     ledger, self.model_strong)
                trace.log("verdict", key=key, upheld=verdict.upheld,
                          corrected=verdict.corrected_value)
                if not verdict.upheld and verdict.corrected_value:
                    board.assert_fact(key, verdict.corrected_value,
                                      board._now_ord + 1, source="debate")
                    adjudications.append(f"{key}: CORRECTED to {verdict.corrected_value}")
                else:
                    adjudications.append(f"{key}: belief UPHELD as stated")
            note = ("Adjudication results for the challenged beliefs:\n"
                    + "\n".join(f"- {a}" for a in adjudications)
                    + "\nAnswer strictly from the belief state below.\n\n")
            proposal = propose(task, board, ledger, self.model_strong,
                               correction_note=note)
            trace.log("reproposal", answer=proposal.answer)
        result = ArmResult(proposal.answer, ledger,
                           [{"role": "trace", "gate": trace.gate,
                             "events": trace.events}])
        return result


def _agora_arm(task: Task, *, seed: int = 0) -> ArmResult:
    return run_agora(task, seed=seed)


ARMS["agora"] = _agora_arm
# Ablations: isolate the value of the gate (sparsity), of debate itself
# (belief board perception only), and of EIG targeting.
ARMS["agora-nogate"] = lambda task, *, seed=0: run_agora(task, seed=seed, gate_mode="always")
ARMS["agora-nodebate"] = lambda task, *, seed=0: run_agora(task, seed=seed, gate_mode="never")
ARMS["agora-noeig"] = lambda task, *, seed=0: run_agora(task, seed=seed, eig_targeting=False)
