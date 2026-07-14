"""Benchmark task families.

Family A — "churn": claim verification under evolving evidence. A seeded
generator produces a stream of dated evidence snippets in which facts get
superseded (values change, sources disagree, some snippets are stale echoes
of retired values). Each task asks whether a claim is true NOW and cites the
deciding evidence. This is the regime where debate has literature support
(verification/factuality) and where a belief-state substrate should shine.

Family B — "multihop": multi-hop QA control. Per Tran & Kiela (2604.02460)
a single agent at equal budget should match or win here — Majalis's gate is
expected to stay mostly CLOSED. We report this honestly as the negative
control that shows the world model knows when debate is not worth tokens.

Deterministic given (family, seed): no network, fully reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Task:
    task_id: str
    family: str
    context: str  # evidence stream / passages, as shown to every arm
    question: str
    gold: str  # canonical short answer ("true"/"false" or an entity)
    meta: dict = field(default_factory=dict)


# --- Family A: evolving-evidence claim verification -------------------------

_ENTITIES = [
    "Meridian Labs", "Halcyon Systems", "Borealis AI", "Vantage Robotics",
    "Cobalt Dynamics", "Skylark Compute", "Nimbus Health", "Aster Materials",
    "Peregrine Energy", "Lumen Biotech", "Cascade Semiconductors", "Onyx Mobility",
]
_ATTRS = {
    "ceo": ["Chen", "Okafor", "Larsson", "Moreau", "Tanaka", "Alvarez", "Novak"],
    "headquarters": ["Austin", "Toronto", "Lyon", "Osaka", "Tallinn", "Porto"],
    "headcount": ["120", "340", "780", "1500", "2200", "4100"],
    "flagship product": ["Atlas", "Kestrel", "Prism", "Argo", "Zephyr", "Nova"],
    "funding round": ["Series A", "Series B", "Series C", "Series D"],
}


_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt_month(month_index: int) -> str:
    """Absolute month index (0 = Jan 2025) -> 'Mar 2025'."""
    return f"{_MONTH_NAMES[month_index % 12]} {2025 + month_index // 12}"


def _churn_stream(rng: random.Random, entity: str, attr: str,
                  values: list[str], start_month: int) -> tuple[list[str], int]:
    """Dated filings for each value in order, with stale echoes of superseded
    values strictly BEFORE the next filing (monotonic date math — the naive
    month-cycling version produced echoes dated after later filings once
    updates exceeded 3, silently corrupting gold)."""
    lines: list[str] = []
    cursor = start_month
    for i, value in enumerate(values):
        filing = cursor
        lines.append(f"[{_fmt_month(filing)}] Filing: {entity}'s {attr} is {value}.")
        cursor += rng.randint(2, 3)
        if i < len(values) - 1:
            for offset in range(1, rng.randint(1, 2) + 1):
                if filing + offset < cursor and rng.random() < 0.8:
                    lines.append(
                        f"[{_fmt_month(filing + offset)}] Blog recap: sources "
                        f"describe {entity}'s {attr} as {value}.")
    return lines, cursor


def _churn_task(rng: random.Random, idx: int, *, min_updates: int = 3,
                max_updates: int = 6, n_distractor_streams: int = 12) -> Task:
    entity = rng.choice(_ENTITIES)
    attr = rng.choice(list(_ATTRS))
    n_updates = rng.randint(min_updates, min(max_updates, len(_ATTRS[attr])))
    values = rng.sample(_ATTRS[attr], k=n_updates)
    snippets, _ = _churn_stream(rng, entity, attr, values, rng.randint(0, 3))

    # Distractor streams: OTHER entities also churn, so extraction and
    # supersession have to work at volume, not just on the asked-about key.
    for _ in range(n_distractor_streams):
        other = rng.choice([e for e in _ENTITIES if e != entity])
        oattr = rng.choice(list(_ATTRS))
        ovals = rng.sample(_ATTRS[oattr], k=rng.randint(2, min(3, len(_ATTRS[oattr]))))
        lines, _ = _churn_stream(rng, other, oattr, ovals, rng.randint(0, 6))
        snippets += lines

    rng.shuffle(snippets)  # presentation order is scrambled; dates carry the truth
    context = "\n".join(snippets)
    current = values[-1]

    if rng.random() < 0.5:
        claimed = current
        gold = "true"
    else:
        claimed = rng.choice(values[:-1])
        gold = "false"
    question = (
        f'Claim: "{entity}\'s {attr} is currently {claimed}." '
        "Based on the most recent dated evidence, is this claim true or false? "
        "Answer 'true' or 'false'."
    )
    return Task(
        task_id=f"churn-{idx}",
        family="churn",
        context=context,
        question=question,
        gold=gold,
        meta={"entity": entity, "attr": attr, "n_updates": len(values)},
    )


# --- Family A2: cross-key comparison under churn ------------------------------

def _compare_task(rng: random.Random, idx: int) -> Task:
    """Claim spans TWO churned numeric keys: both supersession chains must be
    resolved correctly, so per-key errors compound — the regime where
    targeted debate on the weaker belief should pay."""
    e1, e2 = rng.sample(_ENTITIES, k=2)
    snippets: list[str] = []
    finals: list[int] = []
    for entity in (e1, e2):
        n_updates = rng.randint(3, 6)
        values = rng.sample(_ATTRS["headcount"], k=n_updates)
        lines, _ = _churn_stream(rng, entity, "headcount", values, rng.randint(0, 3))
        snippets += lines
        finals.append(int(values[-1]))
    for _ in range(10):
        other = rng.choice([e for e in _ENTITIES if e not in (e1, e2)])
        oattr = rng.choice([a for a in _ATTRS if a != "headcount"])
        ovals = rng.sample(_ATTRS[oattr], k=rng.randint(2, min(3, len(_ATTRS[oattr]))))
        lines, _ = _churn_stream(rng, other, oattr, ovals, rng.randint(0, 6))
        snippets += lines
    rng.shuffle(snippets)

    truly_bigger = finals[0] > finals[1]
    claim_bigger = rng.random() < 0.5
    gold = "true" if claim_bigger == truly_bigger else "false"
    question = (
        f'Claim: "{e1}\'s current headcount is '
        f'{"larger" if claim_bigger else "smaller"} than {e2}\'s current '
        f'headcount." Based on the most recent dated evidence for each '
        "company, is this claim true or false? Answer 'true' or 'false'."
    )
    return Task(
        task_id=f"compare-{idx}",
        family="compare",
        context="\n".join(snippets),
        question=question,
        gold=gold,
        meta={"entities": [e1, e2], "finals": finals},
    )


# --- Family B: multi-hop control ---------------------------------------------

def _multihop_task(rng: random.Random, idx: int) -> Task:
    # Two-hop bridge: entity -> attribute -> entity -> attribute.
    e1, e2 = rng.sample(_ENTITIES, k=2)
    ceo = rng.choice(_ATTRS["ceo"])
    city = rng.choice(_ATTRS["headquarters"])
    product = rng.choice(_ATTRS["flagship product"])
    facts = [
        f"{e1}'s flagship product is {product}.",
        f"The {product} project is led from {e1}'s {city} office.",
        f"{e2} is headquartered in {city}.",
        f"{e2}'s ceo is {ceo}.",
    ]
    # Distractors: never about the bridge entities, and never 'headquarters'
    # (a second company planted in the bridge city would make the by-value
    # city->company hop ambiguous; measured 12% of tasks before this guard).
    others = [e for e in _ENTITIES if e not in (e1, e2)]
    distractor_attrs = [a for a in _ATTRS if a != "headquarters"]
    for _ in range(rng.randint(4, 6)):
        other = rng.choice(others)
        oattr = rng.choice(distractor_attrs)
        facts.append(f"{other}'s {oattr} is {rng.choice(_ATTRS[oattr])}.")
    rng.shuffle(facts)
    question = (
        f"Which person is the ceo of the company headquartered in the same city "
        f"where {e1}'s flagship product project is led from? Answer with the surname only."
    )
    return Task(
        task_id=f"multihop-{idx}",
        family="multihop",
        context="\n".join(facts),
        question=question,
        gold=ceo,
        meta={"hops": 2},
    )


def load_tasks(family: str, n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    maker = {"churn": _churn_task, "compare": _compare_task,
             "multihop": _multihop_task}[family]
    return [maker(rng, i) for i in range(n)]


def grade(task: Task, answer: str) -> bool:
    return task.gold.lower() in answer.strip().lower().split()[-5:] or \
        answer.strip().lower().rstrip(".").endswith(task.gold.lower())
