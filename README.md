# Design for Entropy — Prototype Repository

This repository brings together a set of **software prototypes and experimental models** developed in connection with the book:

**Luca M. Possati, _Design for Entropy. Active Inference and Technology_**  
Cambridge, MA: MIT Press, 2026 https://mitpress.mit.edu/9780262056267/design-for-entropy/

The purpose of this repository is not to present complete, stable, or production-ready software. Rather, it provides **conceptual artifacts and elementary prototypes** designed to clarify, illustrate, and operationalize some of the book’s central arguments.

## Purpose of the Repository

The materials collected here are primarily **illustrative, exploratory, and theory-driven experiments**. Each component should be understood as a **preliminary demonstration** of ideas discussed in the book. These materials are **very elementary prototypes**. They were developed in order to **make the book’s arguments more concrete, accessible, and understandable**, not to provide final applications, engineering benchmarks, or methodologically exhaustive implementations.

## Repository Contents

The repository includes three main groups of materials.

### A. Early versions of the DAU software

This section contains the **earliest iterations of the DAU software**, in their initial and more experimental forms.

These versions make it possible to observe:

- the project’s basic conceptual structure
- the first implementation strategies that were adopted
- the ways in which theoretical hypotheses were translated into minimal computational form
- the gradual emergence of a logic of interaction among model, environment, and interface

### B. More advanced versions of the DAU with graphical dashboard (V2–V3)

The repository also includes later versions of the software, labeled **V2** and **V3**, characterized by greater complexity and the addition of a **graphical dashboard**.

These versions represent a further step beyond the earliest prototypes, especially in terms of:

- process visualization
- interface readability
- graphical representation of variables, states, or model dynamics
- improved accessibility of the theoretical claims through more explicit interfaces

Even so, V2–V3 should not be interpreted as final or complete products. These are still **demonstrative prototypes**, developed to make the visual and operational dimensions of the arguments in _Design for Entropy_ easier to grasp. In that sense, the graphical dashboard serves an epistemic function: it helps make visible what the book develops at the conceptual level.

### C. Example of inversion of the Beck and Ramstead (2025) model https://arxiv.org/abs/2502.21217

A third section of the repository presents an **example of inversion of the Beck and Ramstead (2025) model**.

This material is included as a case study or conceptual exercise intended to:

- show how a theoretical model can be reinterpreted or inverted in an experimental way
- explore some implications of active inference for technological design
- provide concrete support for the book’s theoretical discussion

This example should also be read as a **simplified demonstration**, not as a definitive or exhaustive reconstruction of the original model. Its main value lies in its ability to function as a **bridge between theory and practice**, showing how certain concepts can be tested through minimal formalization.

Every design process targets a specific partition of a system into internal states, external states, and boundary states. What we call the "blanket" is not a causal graph structure in Pearl's sense — it is a dynamical equilibrium: a stable partition where internal and external states stop influencing each other directly, and all their interactions are mediated by the boundary. Beck and Ramstead's algorithm detects this equilibrium in existing systems. What we propose is the inverse operation: design is the process of engineering the inferential conditions — the prior p*(ω) — that make a desired partition the stable attractor of the system's own dynamics. The blanket is not imposed from outside; it is the state the system converges to when the designer's intentions are correctly encoded as constraints on the inference process. Beck and Ramstead detect the blanket as an emergent property of an already organized system. We construct the conditions under which that same property can emerge. The formal definition is identical—T_SZ = T_ZS = 0, I(Z;S|B) → 0—and the metrics we use to verify it are the same ones they use. The only difference is the direction: they move from the system to the blanket, whereas we move from the blanket to the system. But the concept of the blanket we use is the same. This is exactly the sense in which we speak of an “inversion” of their algorithm.

To make these ideas even clearer, three case studies are added: 1) the design of an electrical cable based on active inference; 2) a FEP-designed ideal electric battery; 3) an example of the reduction of the Joule effect in electric conductors via EFE.

## Citation

If you use or discuss these materials, please cite the associated book:

**Luca M. Possati**  
_Design for Entropy. Active Inference and Technology_  
Cambridge, MA: MIT Press, 2026

## Final Note

This repository is best understood as a companion to the book: a space of experimentation in which conceptual arguments are translated into minimal technical forms. Its prototypes are intentionally simple, sometimes highly schematic, and primarily pedagogical in purpose. Precisely for that reason, they are useful: they allow the reader to engage the book’s claims not only at the level of theory, but also through concrete, inspectable, and modifiable examples.
