# Inverse Markov Blanket Battery Design

A computational case study in **inverse design**: a theoretical battery is modeled as an **active inference agent** and optimized to improve **energy efficiency**, **thermal regulation**, and **longevity** by engineering a desired **Markov blanket**.

This repository implements the central idea of Chapter 4 of *Design for Entropy*: instead of using a blanket-detection algorithm only to discover an already existing boundary, we **invert** that logic and use it for **design**. In practical terms, we specify a target statistical boundary and optimize design parameters so that the boundary becomes dynamically real.

## What this repository does

This project combines three pieces:

1. **A battery generative model**
   - The battery is treated as an active inference agent.
   - It senses voltage and temperature.
   - It updates internal beliefs about state of charge, temperature, and health.
   - It acts through power gating and cooling control.

2. **A dynamic Markov blanket detector**
   - Inspired by Beck and Ramstead's dynamic blanket detection framework.
   - Uses time-varying role assignments for microscopic variables.
   - Infers whether each variable behaves as **external**, **blanket**, or **internal** over time.
   - Includes an ELBO-style variational convergence check.

3. **An inverse-design loop**
   - Instead of only detecting a blanket from data, the code places priors on the desired future blanket.
   - Design parameters are optimized so that the resulting system better realizes that blanket.
   - The objective balances blanket quality, energetic performance, longevity, and service.

## Core idea

The main conceptual move is this:

- **Detection** asks: *Given observed dynamics, what Markov blanket is present?*
- **Inverse design** asks: *What interventions and parameters should we choose so that a desired Markov blanket emerges?*

In this case study, the designed object is a **theoretical battery** whose statistical boundary is organized so that it can maintain its own operational integrity while interacting with load, ambient conditions, and charging conditions.

## Battery interpretation

The battery is modeled with the following roles:

- **Internal states**
  - `electrochemical_core`
  - `bms_internal_model`

- **Blanket states**
  - `voltage_sensor`
  - `temperature_sensor`
  - `power_gate`
  - `cooling_actuator`

- **External states**
  - `load_environment`
  - `ambient_environment`
  - `charger_environment`

The goal is not to maximize raw output at all costs. The battery instead tries to remain within a viable operating region while balancing:

- energy efficiency,
- thermal stability,
- state-of-charge regulation,
- health preservation,
- service delivery,
- and the clarity of the internal / blanket / external partition.

## Repository contents

```text
.
├── inverse_markov_blanket_battery_improved_final_nondeps.py
├── outputs/
│   ├── inverse_markov_blanket_battery_rigorous_report.json
│   ├── inverse_markov_blanket_battery_rigorous_summary.txt
│   └── inverse_markov_blanket_battery_rigorous.png
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── CITATION.cff
```

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`
- `joblib` *(optional; if unavailable, the script automatically falls back to serial execution)*

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Quick start

Run the main script:

```bash
python inverse_markov_blanket_battery_improved_final_nondeps.py
```

By default, outputs are written to an `outputs/` directory next to the script.

### Expected outputs

The script generates:

- `inverse_markov_blanket_battery_rigorous_report.json`
- `inverse_markov_blanket_battery_rigorous_summary.txt`
- `inverse_markov_blanket_battery_rigorous.png`

It also prints a concise summary to the terminal, including:

- baseline vs. optimized objective,
- blanket score,
- energy score,
- longevity score,
- service score,
- direct internal-external coupling,
- blanket clarity,
- screening-off score,
- efficiency,
- final health,
- max temperature,
- inferred optimized blanket nodes.

## Optimization backends

The repository currently supports two optimization modes:

### Default: gradient-based SPSA + Adam-style moments

This is the default because it is much more evaluation-efficient than naive random search and works well for this black-box objective.

```bash
python inverse_markov_blanket_battery_improved_final_nondeps.py
```

### Optional: SciPy differential evolution

You can switch to a global black-box optimizer with environment variables:

```bash
BATTERY_OPTIMIZER=de BATTERY_OPT_MAXITER=2 BATTERY_OPT_POPSIZE=4 \
python inverse_markov_blanket_battery_improved_final_nondeps.py
```

## Configuration

The script supports the following environment variables:

- `BATTERY_OUTPUT_DIR`  
  Override the output directory.

- `BATTERY_OPTIMIZER`  
  Optimization backend. Supported values:
  - `spsa` (default)
  - `de`

- `BATTERY_OPT_MAXITER`  
  Number of optimizer iterations.

- `BATTERY_OPT_POPSIZE`  
  Population size for differential evolution.

Example:

```bash
BATTERY_OUTPUT_DIR=./my_results BATTERY_OPT_MAXITER=8 \
python inverse_markov_blanket_battery_improved_final_nondeps.py
```

## How to read the results

The most important outputs are not just raw performance metrics, but **structural** ones:

- **Direct internal-external coupling** should go down.
- **Screening-off score** should go up.
- **Blanket score** should go up.
- The inferred blanket nodes should align with the intended battery boundary.

This means the optimized design is not merely “more efficient.” It is becoming **more agent-like** in the specific sense used in the active inference and Markov blanket literature: its interaction with the environment is increasingly mediated through a coherent statistical boundary.

## Scientific scope

This project is a **theoretical and computational proof of concept**.

It is designed to show that the Chapter 4 inversion can be made explicit in code. It is **not** a validated electrochemical battery simulator and should **not** be treated as engineering guidance for real battery design.

The current implementation uses a pedagogical battery model rather than a full electrochemical model such as PyBaMM. That makes it useful for conceptual work, methodological demonstrations, and theory-building, but not yet for physically realistic validation.

## Limitations

- The battery dynamics are intentionally simplified.
- The generative model is pedagogical rather than experimentally calibrated.
- The blanket detector is variational and approximate.
- The case study demonstrates conceptual plausibility, not laboratory readiness.

## Suggested roadmap

A natural next step would be to:

1. replace the toy battery dynamics with a PyBaMM-based electrochemical model,
2. preserve the inverse-design blanket objective,
3. compare multiple blanket priors and operating regimes,
4. benchmark optimization backends,
5. and validate whether the inferred blanket structure remains stable under more realistic physics.

## Citation

If you use this repository, please cite both the underlying conceptual sources and this implementation.

A starter citation file is included as `CITATION.cff`.

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

## Acknowledgments

This codebase is inspired by:

- Luca M. Possati, *Design for Entropy*, Chapter 4.
- Jeff Beck and Maxwell J. D. Ramstead, *Dynamic Markov Blanket Detection for Macroscopic Physics Discovery*.

## Disclaimer

This repository is for research, theory development, and computational experimentation. It is not medical, industrial, or safety-critical battery engineering advice.
