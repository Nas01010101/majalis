"""Session streams: dated evidence arriving over time, questions interleaved.

This is the deployment-shaped evaluation. A single agent has to re-read the
entire stream-so-far for every question (O(stream) input tokens per
question); a society with a persistent belief board pays perception once per
batch and answers from the compact board (O(board) per question). Accuracy
is graded identically; the cost ledger shows the structural difference.

Ground truth lives in the generator (a dict of current values), so gold is
consistent by construction — including under cross-batch supersession and
stale echoes emitted after the superseding filing arrived (arrival order is
NOT date order, exactly like real feeds).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .tasks import _ATTRS, _ENTITIES, Task, _fmt_month


@dataclass
class SessionEvent:
    kind: str  # "evidence" | "question"
    lines: list[str] = field(default_factory=list)  # evidence batches
    task: Task | None = None  # questions


def make_session(seed: int, *, n_steps: int = 8, keys_per_step: int = 4,
                 questions_per_step: int = 2) -> list[SessionEvent]:
    rng = random.Random(seed)
    truth: dict[tuple[str, str], tuple[str, int]] = {}  # (entity, attr) -> (value, month)
    retired: dict[tuple[str, str], list[tuple[str, int]]] = {}
    events: list[SessionEvent] = []
    month = 0
    q_idx = 0

    for _step in range(n_steps):
        batch: list[str] = []
        for _ in range(keys_per_step):
            if truth and rng.random() < 0.6:
                entity, attr = rng.choice(list(truth))  # update an existing key
            else:
                entity, attr = rng.choice(_ENTITIES), rng.choice(list(_ATTRS))
            old = truth.get((entity, attr))
            candidates = [v for v in _ATTRS[attr] if not old or v != old[0]]
            value = rng.choice(candidates)
            filing_month = month + rng.randint(0, 2)
            if old and old[1] >= filing_month:
                filing_month = old[1] + 1
            batch.append(f"[{_fmt_month(filing_month)}] Filing: {entity}'s "
                         f"{attr} is {value}.")
            if old:
                retired.setdefault((entity, attr), []).append(old)
                # Stale echo of the retired value, dated before the new filing
                # but ARRIVING in the same batch (or later — see below).
                if rng.random() < 0.7 and old[1] + 1 <= filing_month - 1:
                    echo_month = rng.randint(old[1] + 1, filing_month - 1)
                    batch.append(f"[{_fmt_month(echo_month)}] Blog recap: sources "
                                 f"describe {entity}'s {attr} as {old[0]}.")
            truth[(entity, attr)] = (value, filing_month)
        # Late echoes: a retired value resurfacing batches after its retirement.
        if retired and rng.random() < 0.5:
            (entity, attr), history = rng.choice(list(retired.items()))
            old_value, old_month = rng.choice(history)
            cur_month = truth[(entity, attr)][1]
            if old_month + 1 < cur_month:
                echo_month = rng.randint(old_month + 1, cur_month - 1)
                batch.append(f"[{_fmt_month(echo_month)}] Industry note: "
                             f"{entity}'s {attr} remains {old_value}.")
        rng.shuffle(batch)
        events.append(SessionEvent("evidence", lines=batch))
        month += rng.randint(3, 4)

        # Questions about CURRENT truth, biased toward churned keys.
        askable = list(truth)
        churned = [k for k in askable if k in retired]
        for _ in range(questions_per_step):
            pool = churned if churned and rng.random() < 0.7 else askable
            entity, attr = rng.choice(pool)
            value, _m = truth[(entity, attr)]
            if rng.random() < 0.5:
                claimed, gold = value, "true"
            else:
                # Values can be reused: a retired value may equal the current
                # one again, so 'false' claims must exclude the current value.
                past_vals = [v for v, _ in retired.get((entity, attr), [])
                             if v != value]
                claimed = (rng.choice(past_vals) if past_vals
                           else rng.choice([v for v in _ATTRS[attr] if v != value]))
                gold = "false"
            events.append(SessionEvent("question", task=Task(
                task_id=f"s{seed}-q{q_idx}",
                family="stream",
                context="",  # session arms supply their own memory of the stream
                question=(f'Claim: "{entity}\'s {attr} is currently {claimed}." '
                          "Based on the most recent dated evidence, is this "
                          "claim true or false? Answer 'true' or 'false'."),
                gold=gold,
                meta={"entity": entity, "attr": attr,
                      "churned": (entity, attr) in retired},
            )))
            q_idx += 1
    return events
