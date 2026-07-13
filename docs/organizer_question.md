# Organizer question — Dirac-3 / Phase 3 pipeline-stage requirement

**Status:** Drafted by Claude for Sprint 0. Not yet sent — Mouadh sends
manually via Aqora/Discord per team process.

**Where to send:** Aqora / Discord channel for the QRC Weather Forecasting
Challenge 2026.

---

## Draft question

> Phase 3 requires all three pipeline stages executed on QCi Dirac-3.
> Dirac-3 is a polynomial-optimization EQC and cannot execute gate
> circuits. For QRC teams, does this requirement mean (a) a hybrid stage
> (e.g. readout optimization) on Dirac-3, (b) it is boilerplate and
> simulator/IBM/QuEra execution suffices, or (c) something else? What are
> "the three pipeline stages" and "the three key performance metrics"?

---

## Context for whoever sends this

Our pipeline is: encode (ZZ feature map) → TFIM quantum reservoir
(gate-based, simulated via `lightning.qubit`, optional CUDA-Q kernel) →
classical KRR/Ridge readout. None of these stages map naturally onto
Dirac-3's polynomial-optimization model. We need organizer clarification
before committing engineering time to a Dirac-3 integration that may not
be required, or to the wrong integration point (e.g. treating readout
hyperparameter search as a QUBO/Ising problem solvable on Dirac-3, which
*would* map onto its capabilities, versus a literal requirement to run
gate circuits there, which is not possible).
