"""Benchmark task families.

Family A — "churn": claim verification under evolving evidence. A seeded
generator produces a stream of dated evidence snippets in which facts get
superseded (values change, sources disagree, some snippets are stale echoes
of retired values). Each task asks whether a claim is true NOW and cites the
deciding evidence. This is the regime where debate has literature support
(verification/factuality) and where a belief-state substrate should shine.

Family B — "multihop": multi-hop QA control. Per Tran & Kiela (2604.02460)
a single agent at equal budget should match or win here — Agora's gate is
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


def _churn_task(rng: random.Random, idx: int) -> Task:
    entity = rng.choice(_ENTITIES)
    attr = rng.choice(list(_ATTRS))
    values = rng.sample(_ATTRS[attr], k=min(3, len(_ATTRS[attr])))
    months = ["Jan", "Mar", "May", "Jul", "Sep", "Nov"]
    snippets: list[tuple[int, str]] = []  # (order_key, text)

    for i, value in enumerate(values):
        date = f"{months[i * 2 % len(months)]} 202{5 + i // 3}"
        snippets.append((i * 10, f"[{date}] Filing: {entity}'s {attr} is {value}."))
        # Stale echo: a later snippet repeating an already-superseded value.
        if i < len(values) - 1 and rng.random() < 0.7:
            echo_date = f"{months[(i * 2 + 1) % len(months)]} 202{5 + (i + 1) // 3}"
            snippets.append(
                (i * 10 + 5,
                 f"[{echo_date}] Blog recap: sources describe {entity}'s {attr} as {value}.")
            )

    # Distractor facts about other entities.
    for j in range(rng.randint(3, 5)):
        other = rng.choice([e for e in _ENTITIES if e != entity])
        oattr = rng.choice(list(_ATTRS))
        oval = rng.choice(_ATTRS[oattr])
        snippets.append((100 + j, f"[Jun 2026] Note: {other}'s {oattr} is {oval}."))

    rng.shuffle(snippets)  # presentation order is scrambled; dates carry the truth
    context = "\n".join(text for _, text in snippets)
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
    maker = {"churn": _churn_task, "multihop": _multihop_task}[family]
    return [maker(rng, i) for i in range(n)]


def grade(task: Task, answer: str) -> bool:
    return task.gold.lower() in answer.strip().lower().split()[-5:] or \
        answer.strip().lower().rstrip(".").endswith(task.gold.lower())
