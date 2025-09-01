# Aggregators

Aggregators implement the stepping logic for simulating jump processes.

## Abstract Base Classes

::: jumpax.AbstractAggregator
    options:
        members:
            - init
            - step

::: jumpax.AbstractHybridAggregator
    options:
        members: false

## Pure Jump Aggregators

---

::: jumpax.SSA
    options:
        members: false

---

::: jumpax.SimpleTauLeaping
    options:
        members: false

## Hybrid Aggregators

Hybrid aggregators combine jump processes with continuous dynamics (ODEs/SDEs) using diffrax.

---

::: jumpax.HybridSSA
    options:
        members:
            - __init__

---

::: jumpax.HazardSSA
    options:
        members:
            - __init__
