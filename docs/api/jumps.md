# Jump Problems

Jump problems define the rates and effects of discrete jumps in the system.

## Abstract Base Classes

::: jumpax.AbstractJumpProblem
    options:
        members:
            - init
            - rate
            - affect
            - leap_delta

::: jumpax.AbstractAffect
    options:
        members:
            - init
            - __call__

## Jump Problem Types

---

::: jumpax.ConstantRateJump
    options:
        members:
            - __init__

---

::: jumpax.MassActionJump
    options:
        members:
            - __init__

---

::: jumpax.VariableRateJump
    options:
        members:
            - __init__

---

::: jumpax.StatelessAffect
    options:
        members: false
