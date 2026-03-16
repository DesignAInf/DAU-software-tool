# Emergent Conduit based on FEP

## Overview
This repository contains a 200-step simulation. In it, a designer doesn't begin with a ready-made cable — instead, they build one from scratch using a principled process:

-Start with a model of electron flow. Rather than assuming a fixed structure, the designer works from a probabilistic model of how electrons move through a material.
-Dynamically assign roles to microstates. As the simulation runs, individual microstates (tiny regions of the material) are continuously classified as internal, boundary, or external — forming what's called a Markov blanket.
-Choose interventions intelligently. The designer uses a quantity called expected free energy (EFE) to decide which interventions will most effectively guide the system toward a useful structure. EFE balances exploration (trying new things) with exploitation (reinforcing what works).
-Watch a cable-like structure emerge. Over time, the boundary microstates self-organize into a conduit — a channel that concentrates and directs electron flow rather than letting energy scatter.

The structure pays for itself. The energy cost of the initial interventions is eventually offset by the reduction in energy diffusion the emergent structure provides.

This is connected to the book: Luca M. Possati, __Design for Entropy. Active Inference and Technology__
Cambridge, MA: MIT Press, 2026 https://mitpress.mit.edu/9780262056267/design-for-entropy/

## What the simulation does
The simulation begins with a cloud of electrons moving through a longitudinal domain while external perturbations act transversely on the cloud. At each step, the code:

1. infers dynamic role assignments for every electron (**external, boundary, internal**);
2. updates macrostate statistics for the emerging conduit (**center, width, precision**);
3. evaluates a discrete set of design policies using **expected free energy** over a finite horizon;
4. penalizes policies that increase Joule-like diffusion, radial leakage, role mismatch, uncertainty, and unpaid energy debt versus a passive baseline;
5. applies the selected policy and lets the new conduit geometry emerge from the electron flow itself.

This operationalizes the design-side inversion discussed in your manuscript: the blanket is not merely detected after the fact, but actively shaped through designer priors, role assignments, attention, and intervention.

## Relationship to Beck & Ramstead
The code is directly inspired by the logic of **dynamic Markov blanket detection** associated with Beck & Ramstead:
- microstates receive dynamic role assignments;
- role assignments are updated variationally;
- boundary structure is treated as the key mediator between internal and external dynamics;
- the macro-object is defined by the path statistics of the blanket.

The present code then pushes that logic in the **inverse, design-side direction**: the designer selects interventions so that a conduit-like blanket *emerges* and becomes energetically worthwhile.

## Relationship to the Free Energy Principle
The simulation separates two levels:
- **variational free energy (VFE)** for state and role inference;
- **expected free energy (EFE)** for policy selection over future horizons.

Here, EFE explicitly includes:
- Joule-like diffusion risk,
- radial leakage risk,
- conduit-width mismatch,
- role-composition mismatch,
- ambiguity,
- epistemic value,
- energy debt versus a passive baseline.

This makes the model especially useful for showing a practical design claim: lowering EFE can be tied to lowering energy diffusion *and* to recovering the initial control cost.

## What the visualization shows
The main figure and animation show three things clearly:

1. **Microstate organization**: electrons separate into internal, boundary, and external roles.
2. **Emergent conduit geometry**: a narrow internal transport core forms, wrapped by a boundary shell.
3. **Energy payback**: the designer may spend more energy at first, but then crosses break-even and outperforms a passive baseline.

## Practical result from the default 200-step run
The default run in this repository uses **200 steps** and produces the following result:

- break-even step: **82**
- final cumulative designer Joule: **0.24182578844246339**
- final cumulative passive Joule: **0.3861354842959562**
- final cumulative saving vs passive: **0.14430969585349301**
- final energy debt: **0.0**
- final EFE-to-Joule correlation: **0.990586740909214**
- final internal fraction: **0.2703862775364967**
- final boundary fraction: **0.49954875609559396**
- final external fraction: **0.23006496538592983**
- final conduit width sigma: **0.1602645901972071**

In practical terms, the run shows that the designer can pay an early energetic cost to shape the blanket, reach **break-even at step 82**, and finish with a lower cumulative dissipative burden than the passive baseline.

## Why this matters for designers and engineers
For a designer, the repository shows that form can be treated as an emergent consequence of inference and intervention rather than as a fixed shell imposed from the outside.

For an engineer, the practical message is even sharper: the best blanket is not the one that merely lowers end-stage dissipation, but the one that lowers dissipation enough to **repay its startup cost** and deliver a net energy benefit.

## Quick start
Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the default 200-step simulation:

```bash
emergent-conduit-fep
```

Run a longer simulation:

```bash
emergent-conduit-fep --steps 5000 --frame_stride 50
```

Or run it as a module:

```bash
python -m emergent_conduit_fep --steps 200
```

## Output files
By default, the code writes output to a local `emergent_conduit_payback_outputs/` folder and saves:
- `history_designer.csv`
- `history_passive.csv`
- `summary.csv`
- `emergent_conduit_payback_snapshot.png`
- `emergent_conduit_payback.gif`
- `emergent_conduit_payback.mp4` (when ffmpeg is available)

## Repository layout
- `src/emergent_conduit_fep/simulation.py` — main simulation and CLI
- `examples/` — example outputs from the 200-step run
- `tests/test_smoke.py` — minimal smoke test

## Notes on scope
This repository is best understood as an **operational, design-side proof of concept**. It is strongly aligned with the Beck-Ramstead blanket framework and with a Fristonian EFE logic, but it should be described as a rigorous simulation study rather than as a general theorem about blanket invertibility.
