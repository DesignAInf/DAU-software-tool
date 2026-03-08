# Torus Emergence via Inverted DMBD

**Beck & Ramstead (2025) — arXiv:2502.21217**

A simulation of 2000 microscopic elements self-organizing into a toroidal Markov blanket, driven by the inversion of the Dynamic Markov Blanket Detection (DMBD) algorithm. Color encodes assignment certainty: all elements begin grey (maximum entropy), and the toroidal shell emerges visually as `q(ω)` sharpens over iterations.

**Introduction**

We built a simulation that shows how an object emerges from nothing.

_The starting point_
Imagine 4000 dots scattered randomly in space. They don't know what they are, they don't know where they need to go. Each dot has an open question: am I inside, outside, or on the boundary of something?

_What happens when you press RUN_
Each dot tries to answer that question by looking around itself. As it answers, it moves toward the position that matches its answer. And by moving, the answer becomes more certain. It's a self-reinforcing cycle.
The color you see emerging is not decorative — it's certainty. Grey means "I don't know yet what I am." Gold, blue, red means "I know."

_The torus_
The shape that emerges — the donut — is not drawn by hand. It is the result of a designer prior: a mathematical preference embedded in the system. The dots don't know they are supposed to form a torus, but the prior pushes them in that direction. It's like the difference between telling someone "make a donut" and giving them a recipe that, followed step by step, inevitably produces a donut.

_The three roles_
Each dot eventually assigns itself to one of three roles:

Z internal (blue) — inside the tube
S external (red) — outside
B blanket (gold) — on the surface, the boundary between inside and outside

The blanket is what matters. It is the boundary that separates the system from its environment. In Friston's Free Energy Principle, every system that maintains its own identity has a Markov blanket — a boundary that mediates all interactions between inside and outside.

_The verification_
The charts at the bottom are not ornamental. The most important one is I(Z;S|B) — it measures whether the boundary actually works. If it falls toward zero, it means the inside and the outside no longer communicate directly: all communication passes through the blanket. This is exactly the mathematical definition of a boundary that works.

_The connection to design_
Beck and Ramstead use this algorithm to detect boundaries that already exist. We inverted it: we use the same algorithm to create boundaries. This is the central claim of your manuscript — that design is the engineering of boundaries, and that it can be formalized mathematically. The three sliders (Precision, Curiosity, Embedding) are three different ways a designer intervenes in that process.

---

## Contents

1. [The Core Idea: Detection vs. Inversion](#1-the-core-idea-detection-vs-inversion)
2. [The DMBD Algorithm (Beck & Ramstead 2025)](#2-the-dmbd-algorithm-beck--ramstead-2025)
3. [Inversion: From Detection to Design](#3-inversion-from-detection-to-design)
4. [The Equations — Line by Line](#4-the-equations--line-by-line)
   - [4.1 Variational Posterior](#41-variational-posterior-factorization)
   - [4.2 Designer Prior](#42-designer-prior)
   - [4.3 ATTEND Step](#43-attend-step-eq-26)
   - [4.4 INFER Step](#44-infer-step-eq-27)
   - [4.5 Lagrange Multiplier](#45-lagrange-multiplier-eq-13)
   - [4.6 Expected Free Energy](#46-expected-free-energy-efe)
   - [4.7 EFE Forces](#47-efe-forces--physical-implementation)
   - [4.8 Euler-Maruyama Physics](#48-euler-maruyama-langevin-dynamics)
   - [4.9 ELBO](#49-elbo--variational-free-energy)
5. [The Three Design Levers](#5-the-three-design-levers)
6. [Visual Encoding: Color as Certainty](#6-visual-encoding-color-as-certainty)
7. [The Torus as Target](#7-the-torus-as-target)
8. [Convergence and Reification](#8-convergence-and-reification)
9. [Quickstart](#9-quickstart)
10. [References](#10-references)

---

## 1. The Core Idea: Detection vs. Inversion

The Beck & Ramstead (2025) paper introduces an algorithm — DMBD — for *detecting* Markov blankets in physical systems. Given a dataset of particle trajectories, DMBD discovers which particles form the boundary (blanket) between an internal system and its environment.

This simulation runs the algorithm **in the opposite direction**.

| Direction | Input | Output | Purpose |
|---|---|---|---|
| **Detection** (original) | Particle data Y | Blanket assignment ω* | Discover an existing boundary |
| **Inversion** (this simulation) | Target geometry p*(ω) | Particle configuration X | *Create* a new boundary |

The inversion corresponds to what the manuscript calls **design as the engineering of Markov blankets**: instead of finding a blanket that already exists in data, we specify what blanket we want and let the system self-organize to produce it.

The target here is a torus. It is never given as a set of coordinates. It is given only as a functional prior — a probability distribution over roles — and the 2000 elements find their way to the toroidal surface through VBEM inference and EFE forces alone.

---

## 2. The DMBD Algorithm (Beck & Ramstead 2025)

A Markov blanket partitions elements into three roles:

- **Z** — internal states (inside the boundary, shielded from environment)
- **B** — blanket states (the boundary itself: sensory + active states)
- **S** — external states (environment, outside the boundary)

The blanket condition requires that internal and external states are **conditionally independent given the blanket**:

```
p(Z, S | B) = p(Z | B) · p(S | B)
```

In the transition matrix formalism (Eq. 21 of Beck & Ramstead), this is enforced as:

```
T_SZ = T_ZS = 0
```

meaning no direct causal flow between Z and S — all influence passes through B.

DMBD finds this partition using **Variational Bayes Expectation-Maximization (VBEM)**. At each iteration, it alternates between:

- **ATTEND**: updating the variational posterior `q(ω)` over role assignments
- **INFER**: updating macroscopic latent variables (centroids, covariances of each role)

The algorithm converges when the **ELBO** (Evidence Lower BOund) stops increasing — meaning `q(ω)` is as consistent as possible with both the data and the Markov blanket constraint.

---

## 3. Inversion: From Detection to Design

The inversion introduces two modifications to the standard DMBD pipeline:

**Modification 1: Replace the data likelihood with a designer prior.**

In standard DMBD, `q(ω)` is updated to be consistent with observed data Y. In the inversion, we instead push `q(ω)` toward a *designer-specified* target distribution `p*(ω)` that encodes the desired geometry. This prior replaces (or augments) the data likelihood in the ATTEND step.

**Modification 2: Add EFE forces to the physics.**

In standard DMBD, particle positions are given (they are the data). In the inversion, positions are dynamic. Elements experience forces derived from their Expected Free Energy `G(π)` — a quantity that measures how far the current configuration is from the target. Elements whose assignment is uncertain and inconsistent with the prior receive strong forces pushing them toward their designated zone.

The result is a closed loop:

```
positions → q(ω) → EFE forces → new positions → q(ω) → ...
```

This loop is the inversion. It does not converge to a fixed point in general — it converges to a *regime* where the particle configuration is statistically consistent with the designer's prior, and the role assignments are as certain as the physics allows.

---

## 4. The Equations — Line by Line

### 4.1 Variational Posterior Factorization

Following Eq. 25 of Beck & Ramstead, the full variational posterior factorizes as:

```
q(Z, B, S, ω, Θ) = q_{ZBS}(Z,B,S) · q_ω(ω) · q_Θ(Θ)
```

In this simulation, `Θ` (the macroscopic parameters — centroids and covariances of each role) are treated as point estimates updated deterministically. The only stochastic quantity tracked per element is `q(ωᵢ)` — a categorical distribution over the three roles.

At initialization:

```
q(ωᵢ = k) = 1/3   for all i, k ∈ {S, B, Z}
```

This is the maximum entropy state: `H[q] = log 3 ≈ 1.099`. No blanket exists yet.

### 4.2 Designer Prior

The designer's prior `p*(ωᵢ)` encodes the target geometry. For a torus with major radius `R` and minor radius `r`, the signed distance function is:

```
d(xᵢ) = sqrt( (sqrt(xᵢ² + zᵢ²) − R)² + yᵢ² ) − r
```

where `d < 0` means inside the tube, `d = 0` means on the surface, `d > 0` means outside.

Role propensities are:

```
prop_B(i) = exp(−½ · d(xᵢ)² / σ_B²)              [near the surface, d ≈ 0]
prop_Z(i) = 1 / (1 + exp( d(xᵢ) / 0.5))           [inside the tube, d < 0]
prop_S(i) = 1 / (1 + exp(−d(xᵢ) / 0.5))           [outside the tube, d > 0]
```

The shell band `σ_B` tightens with convergence and precision:

```
σ_B = clamp( (1.05 − conv · 0.60 − precision · 0.25) · r_minor,  0.12,  r_minor )
```

The prior is then:

```
p*(ωᵢ = B) ∝ exp(γ · prop_B(i))
p*(ωᵢ = Z) ∝ exp(γ · prop_Z(i) · 0.72)
p*(ωᵢ = S) ∝ exp(γ · prop_S(i) · 0.55)
```

with normalization, and concentration parameter:

```
γ = γ₀ + 4 · precision
```

where `γ₀ = 2.2` and `precision ∈ [0,1]` is the Precision Crafting lever. Higher γ makes the prior sharper — elements near the surface receive a much stronger signal to become B.

**Why this is the inversion:** In standard DMBD, `p*` is either absent or uniform. Here it encodes the designer's intent. The entire geometry emerges from this prior — no coordinates are ever explicitly assigned.

### 4.3 ATTEND Step [Eq. 26]

The variational update for `q(ωᵢ)` is:

```
log q(ωᵢ = k)  ∝  log p(yᵢ | ωᵢ = k, macro)  +  λ · log p*(ωᵢ = k)
```

This is the core of the inversion. The first term is the **emission likelihood** — how consistent is element `i`'s current position with the macroscopic description of role `k`? The second term is the **designer prior** — how consistent is role `k` with the target geometry?

The parameter `λ(t)` (see §4.5) controls the relative weight of the two terms.

**Emission likelihood.** The macroscopic latent for each role is the mean signed distance:

```
d̄_k = E_{q(ωᵢ=k)}[d(xᵢ)]  =  Σᵢ q(ωᵢ=k) · d(xᵢ)  /  Σᵢ q(ωᵢ=k)
```

The emission model is Gaussian on signed distance:

```
log p(yᵢ | ωᵢ = k, macro)  =  −½ · (d(xᵢ) − d̄_k)² / σ̂_k²
```

where `σ̂_k = Std_{q_k}[d(xᵢ)] + ε` is the empirical standard deviation of signed distance for role `k`.

**Softmax normalization.** Taking the exponential and normalizing:

```
q(ωᵢ = k)  =  exp(log q̃(ωᵢ = k))  /  Σⱼ exp(log q̃(ωᵢ = j))
```

This is computed in log-space with max subtraction for numerical stability.

### 4.4 INFER Step [Eq. 27]

After updating `q(ωᵢ)`, the macroscopic latents are recomputed:

```
d̄_k  =  Σᵢ q(ωᵢ=k) · d(xᵢ)  /  Σᵢ q(ωᵢ=k)

σ̂_k  =  sqrt( Σᵢ q(ωᵢ=k) · (d(xᵢ) − d̄_k)²  /  Σᵢ q(ωᵢ=k) )  +  ε
```

As the torus forms, `d̄_B → 0` (B elements concentrate on the surface where `d = 0`) and `σ̂_B → 0` (B elements become tightly distributed on the surface). These two quantities are shown in the dashboard as live convergence indicators.

The INFER step is the part of DMBD that "reads back" the current state of the system to update the model's belief about what each role looks like. In the inversion, it also feeds back into the EFE forces.

### 4.5 Lagrange Multiplier [Eq. 13]

The Lagrange multiplier `λ(t)` enforces the Markov blanket constraint. In the variational formulation of DMBD (deriving from Jaynes' maximum caliber, §2 of Beck & Ramstead), it appears as:

```
λ(t)  =  λ₀ · (1 + t · η_λ) · max(0.08,  1 − conv · 0.65)
```

with `λ₀ = 0.52`, `η_λ = 0.004`.

Three effects:

1. `λ₀` sets the baseline strength of the prior relative to the likelihood.
2. `(1 + t · η_λ)` grows monotonically — the constraint becomes stronger as the simulation progresses. This is an annealing schedule that prevents the prior from dominating too early, when positions are still chaotic.
3. `(1 − conv · 0.65)` decreases λ as convergence increases — once the blanket is formed, the system no longer needs to be pushed strongly. This prevents over-contraction.

### 4.6 Expected Free Energy (EFE)

The Expected Free Energy per element is the central quantity driving the inversion. Following §2 of Beck & Ramstead (and the active inference literature more broadly):

```
G(π)  =  KL[ q(ωᵢ) ‖ p*(ωᵢ) ]  −  β · H[ q(ωᵢ) ]
       =  Risk  −  β · Epistemic Value
```

**KL divergence (Risk):**

```
KL[ q ‖ p* ]  =  Σₖ q(ωᵢ=k) · log( q(ωᵢ=k) / p*(ωᵢ=k) )
```

This measures how far the current assignment distribution `q` is from the target `p*`. When `q` agrees with `p*`, KL = 0. When they disagree, KL > 0 and the element is "out of place."

**Shannon entropy (Epistemic Value):**

```
H[ q(ωᵢ) ]  =  −Σₖ q(ωᵢ=k) · log q(ωᵢ=k)
```

This is maximum (log 3 ≈ 1.099) when `q` is uniform (element fully unassigned) and zero when `q` is a point mass (element fully assigned to one role). The term `−β · H[q]` in G(π) means that **epistemic uncertainty contributes negatively to free energy** — the system is rewarded for maintaining uncertainty. This is the active inference sense in which curiosity has instrumental value.

**β** is the Curiosity Sculpting lever: `β = 0.18 + 0.60 · curiosity`. High β means the system actively avoids premature certainty and explores more before committing.

**G(π) as a force generator.** When `G(π) > 0`, the element is not yet consistent with the target — it should move. The physical force derived from G is:

```
F_EFE  ∝  −λ(t) · G(π) · ∇_x[ dist_to_target_zone ]
```

This moves each element along the gradient of its assigned-zone distance, weighted by how strongly it needs to move (G) and the current constraint strength (λ).

### 4.7 EFE Forces — Physical Implementation

The gradient of the torus signed distance function with respect to position is:

```
∂d/∂xᵢ  =  (d_XZ / |d_total|) · x̂_XZ
∂d/∂yᵢ  =  yᵢ / |d_total|
∂d/∂zᵢ  =  (d_XZ / |d_total|) · ẑ_XZ
```

where `d_XZ = sqrt(xᵢ² + zᵢ²) − R` is the radial deviation from the tube centre ring, and `x̂_XZ`, `ẑ_XZ` are unit vector components in the XZ plane.

Forces by role (scaled by `forceScale = 0.20 + 0.35 · precision`):

```
F_B  =  −d(xᵢ) · λ · G · q_B · 1.8 · ∇d   [toward surface: restoring force to d=0]
F_Z  =  +λ · G · q_Z · 0.55 · ∇d            [inward: push against gradient]
F_S  =  −λ · G · q_S · 0.40 · ∇d            [outward: push along gradient]
```

Note that `F_B` is a restoring force proportional to `d` — like a spring pulling B elements toward the surface `d = 0`. This is the key mechanism that makes the torus form: B elements experience stronger forces the further they are from the surface, regardless of direction.

Contact repulsion prevents collapse: for randomly sampled pairs `(i, j)`:

```
F_repulsion  =  K_excl · (R_excl − |xᵢ − xⱼ|) / |xᵢ − xⱼ|  · (xᵢ − xⱼ)   if |xᵢ−xⱼ| < R_excl
```

with `K_excl = 0.55`, `R_excl = 0.22`. Only ~4500 pairs are sampled per step (instead of all N² = 4,000,000) for computational tractability.

### 4.8 Euler-Maruyama Langevin Dynamics

The full stochastic differential equation governing each element is:

```
dxᵢ  =  F_total(xᵢ, q, t) · dt  +  σ(t) · dWᵢ
```

where `dWᵢ` is a Wiener process increment. In discrete form (Euler-Maruyama):

```
vᵢ(t+dt)  =  α · vᵢ(t)  +  F_total · dt
xᵢ(t+dt)  =  xᵢ(t)  +  vᵢ(t+dt) · dt  +  σ(t) · √dt · ηᵢ
```

with `ηᵢ ~ N(0,1)`, `α = 0.80` (velocity damping), `dt = 0.018`.

The noise amplitude `σ(t)` implements Prediction Embedding (annealing):

```
σ(t)  =  clamp(  σ₀ · (1 − 0.68 · embedding) · exp(−t · 0.0018 · embedding)  +  0.006,   0.005,  0.06  )
```

with `σ₀ = 0.040 − 0.010 · precision`. At `embedding = 0`, noise is constant. At `embedding = 1`, noise decays exponentially toward the minimum floor of 0.006 — the system becomes progressively more deterministic as it converges.

The Langevin noise term has two roles: it prevents the system from freezing in suboptimal configurations (thermodynamic exploration), and it makes the simulation physically faithful to the stochastic dynamics that underpins the Free Energy Principle formulation in Beck & Ramstead.

### 4.9 ELBO — Variational Free Energy

The Evidence Lower BOund measures overall quality of the variational approximation:

```
ELBO  =  E_q[ log p*(ω) ]  −  KL[ q(ω) ‖ p*(ω) ]
       =  Σᵢ Σₖ q(ωᵢ=k) · [ log p*(ωᵢ=k)  −  log q(ωᵢ=k) ]
```

normalized per element. The ATTEND step is equivalent to maximizing this quantity with respect to `q`. A rising ELBO means the role assignments are becoming more consistent with the designer's prior — the blanket is crystallizing.

Note the relationship to standard variational inference: maximizing ELBO is equivalent to minimizing KL[q ‖ p*], which is equivalent to minimizing the variational free energy F = −ELBO. The system self-organizes to reduce free energy, and the geometric form (the torus) is what minimal free energy looks like when the prior encodes toroidal geometry.

---

## 5. The Three Design Levers

These correspond to the three mechanisms identified in the manuscript as central to design understood as the engineering of Markov blankets.

### Precision Crafting

**What it controls:** The concentration of the designer's prior, γ.

**Equation:** `γ = γ₀ + 4 · precision`, where `γ₀ = 2.2`. Also tightens the shell band: `σ_B ∝ (1 − 0.25 · precision)`.

**Effect on G(π):** A higher γ makes `p*(ω)` sharper — the KL divergence grows faster when elements are misplaced. This amplifies the Risk term in `G(π) = KL − β·H`, generating stronger EFE forces.

**Visual effect:** Blanket forms faster and more sharply. Elements near the surface commit to B quickly; elements far from the surface are pushed away decisively.

**Tradeoff:** High precision risks freezing the system in a suboptimal configuration before the global toroidal structure has time to emerge. The system may form local patches of B rather than a coherent shell.

**Manuscript connection:** Precision crafting corresponds to sharpening the precision weighting of the generative model — making the model more confident about what it expects, at the cost of flexibility.

### Curiosity Sculpting

**What it controls:** The epistemic weight β in the EFE.

**Equation:** `G(π) = KL[q‖p*] − β·H[q]`, with `β = 0.18 + 0.60 · curiosity`.

**Effect:** A higher β means `−H[q]` contributes more negatively to `G(π)`. Elements with high assignment entropy (uncertain, exploring) have *lower* free energy than elements with low entropy (committed, settled). The system is rewarded for maintaining uncertainty — it explores more space before committing to a role.

**Visual effect:** At high curiosity, points remain grey longer — they stay uncertain as they wander. The gold shell appears later but may emerge more uniformly distributed over the toroidal surface.

**Tradeoff:** Too much curiosity prevents convergence altogether: elements never commit, and ELBO plateaus at a low value.

**Manuscript connection:** Curiosity sculpting corresponds to controlled epistemic uncertainty — designing artifacts or environments that provoke exploration rather than immediate exploitation. The curiosity term is directly related to the epistemic value component of active inference policies.

### Prediction Embedding

**What it controls:** The signal-to-noise ratio of the dynamics — specifically the Langevin noise amplitude σ(t).

**Equation:** `σ(t) = σ₀ · (1 − 0.68 · embedding) · exp(−t · 0.0018 · embedding) + σ_floor`

**Effect:** High embedding means low noise and exponential annealing. Elements trust the EFE forces and follow them with less thermal deviation. The dynamics become progressively more deterministic as the system converges.

**Visual effect:** Trajectories are smoother. The shell sharpens faster once it begins to form, with less jitter. Points that have committed to a role stay committed.

**Tradeoff:** Low noise means the system cannot escape local minima. If elements become trapped in a bad configuration early (e.g., a lopsided partial shell), high embedding will freeze them there.

**Manuscript connection:** Prediction embedding corresponds to externalizing generative model predictions into the physical structure of the environment — making the environment itself carry the model's expectations. In the simulation, this manifests as reducing the stochasticity that would otherwise allow elements to deviate from their predicted trajectories.

---

## 6. Visual Encoding: Color as Certainty

The most important visual decision in this simulation is that **color encodes certainty, not role**.

For each element `i`, the certainty is:

```
cert(i)  =  (max_k q(ωᵢ=k) − 1/3)  /  (2/3)   ∈ [0, 1]
```

When `q` is uniform (maximum entropy, no assignment), `max_k q = 1/3` and `cert = 0`. When `q` is a point mass on one role, `max_k q = 1` and `cert = 1`.

Color is computed as:

```
hue        =  role color  (0° red for S,  38° gold for B,  210° blue for Z)
saturation =  cert² · 70%
lightness  =  12% + cert² · 22%
alpha      =  0.10 + cert² · 0.75
```

Using `cert²` (rather than `cert`) creates a sharper perceptual onset: elements remain visually grey until certainty exceeds roughly 0.4, then transition quickly to full saturation. This makes the *emergence* of the blanket visible as a sudden brightening of the toroidal surface, rather than a gradual fade.

The glow effect around high-certainty blanket elements (`cert > 0.62`, role = B) is a radial gradient that appears only as the shell forms — making the Markov boundary literally luminous at the moment of reification.

---

## 7. The Torus as Target

The torus was chosen for three reasons.

First, it is topologically non-trivial — unlike a sphere, it has a hole. This means the blanket condition (internal states shielded from external states) requires a fundamentally different spatial arrangement: the internal zone Z is the interior of the tube, not a simple interior of a convex hull. The system must discover this topology from the prior alone.

Second, the torus signed distance function `d(x) = sqrt((sqrt(x²+z²) − R)² + y²) − r` has a clean analytical gradient, making the EFE force computation exact rather than numerically approximated.

Third, it is more visually distinctive than a sphere. As the simulation progresses, the hole in the torus becomes visible as a region of low element density, surrounded by the gold shell. This makes convergence perceptually obvious.

The torus parameters are `R_major = 2.6` (distance from Y-axis to tube centre) and `R_minor = 0.85` (tube radius). The wireframe outline of the target geometry is always visible in the canvas as a faint gold grid — it is present from the start, before any elements reach it, emphasizing that the target is given only as a prior, not as a physical scaffold.

---

## 8. Convergence and Reification

The system is considered to have **reified** the toroidal blanket when:

```
conv > 0.68   AND   iter > 30
```

where convergence is defined as the fraction of B-assigned elements (those with `q_B > 0.45`) that are within `|d(xᵢ)| < 0.55` of the torus surface:

```
conv  =  | { i : q_B(i) > 0.45  AND  |d(xᵢ)| < 0.55 } |  /  | { i : q_B(i) > 0.45 } |
```

The threshold `|d| < 0.55` is approximately 65% of `R_minor = 0.85` — so elements must be within the inner 65% of the tube radius to count as converged.

At reification, the `BLANKET REIFIED` indicator appears in the header, and the log records the event. The system continues to evolve — convergence is a regime, not a fixed point — but the toroidal Markov partition is now stable enough to function as a boundary in the formal sense.

The key indicators to watch during convergence:

- **ELBO** rises monotonically (or near-monotonically) — assignments becoming more consistent with the prior
- **H[q]** falls from log(3) ≈ 1.099 toward 0 — assignments sharpening
- **KL[q‖p*]** falls — q converging toward p*
- **d̄_B → 0** — B elements concentrating on the surface
- **Color** transitions from grey to gold on the toroidal surface

---

## 9. Quickstart

```bash
# Unzip in Finder (double-click), then drag folder to Desktop
cd ~/Desktop/torus-dmbd
npm install
npm run dev
# Open http://localhost:5173
```

Press **▶ RUN**. All elements begin grey. Within ~50 iterations, faint colour begins to appear near the torus surface. By ~150–300 iterations (depending on lever settings), a visible gold shell forms.

**Recommended starting levers:**
- Precision: 0.55 (moderate — strong enough to guide, loose enough to explore)
- Curiosity: 0.35 (allows some exploration without preventing convergence)
- Embedding: 0.50 (moderate annealing)

**If the system is stuck** (no colour emerging after 200+ iterations): try increasing Precision to 0.80. If the shell forms in patches but won't close: try decreasing Curiosity to 0.15 and increasing Embedding to 0.70.

---

## 10. References

**Primary:**
- Beck, J. & Ramstead, M.J.D. (2025). *Dynamic Markov Blanket Detection for Macroscopic Physics Discovery*. arXiv:2502.21217.

**Free Energy Principle & Active Inference:**
- Friston, K. et al. (2023). *Path integrals, particular kinds, and strange things*. Physics of Life Reviews.
- Friston, K. (2019). *A free energy principle for a particular physics*. arXiv:1906.10184.
- Parr, T., Pezzulo, G. & Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

**Maximum Caliber (underpins DMBD's variational formulation):**
- Jaynes, E.T. (1957). *Information theory and statistical mechanics*. Physical Review, 106(4).
- Sakthivadivel, D. (2022). *Towards a geometry and analysis for Bayesian mechanics*. arXiv:2204.11900.

**Structure Learning (Bayesian model reduction used in DMBD):**
- Smith, R. et al. (2020). *Structure learning as an inferential process*. arXiv:2007.10936.

**Design Theory (manuscript context):**
- Manuscript: *Design as the Engineering of Markov Blankets*. MIT Press (forthcoming, uncorrected proofs).
