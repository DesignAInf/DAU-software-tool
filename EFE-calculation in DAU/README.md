# Expected Free Energy (EFE) Calculation for DAU System

## Overview
This repository provides a Python implementation to compute the **Expected Free Energy (EFE)** for agents in a **Designer-Artifact-User (DAU)** active inference system. The implementation is based on the **pymdp** library and follows active inference principles to evaluate the optimality of policies taken by each agent.

## Features
- Computes **Expected Free Energy (EFE)** for each agent (Designer, Artifact, User).
- Evaluates both **instrumental** (goal-driven) and **epistemic** (information-seeking) components of decision-making.
- Uses the **pymdp** library for Bayesian inference and active inference modeling.
- Supports **visualization** of EFE values for different policies.

## Installation
Ensure you have Python 3 installed, then install the required dependencies:
```bash
pip install pymdp numpy matplotlib
```

## Usage

### 1 Define DAU Agents
Modify or add the following code in `dau_active_inference.py` to create agents:
```python
from pymdp.agent import Agent

# Define model dimensions
num_states = [3, 2]  # Example state space dimensions
num_observations = [3, 2]  # Example observation dimensions
num_controls = [2, 2]  # Example control dimensions

# Create DAU agents
designer_agent = Agent(num_states=num_states, num_obs=num_observations, num_controls=num_controls)
artifact_agent = Agent(num_states=num_states, num_obs=num_observations, num_controls=num_controls)
user_agent = Agent(num_states=num_states, num_obs=num_observations, num_controls=num_controls)

# Store them in a dictionary
dau_agents = {
    "Designer": designer_agent,
    "Artifact": artifact_agent,
    "User": user_agent
}
```

### 2 Compute Expected Free Energy (EFE)
Call the EFE computation function after defining agents:
```python
from calculate_efe_dau import compute_efe_for_dau_system

# Compute EFE for DAU agents
efe_results = compute_efe_for_dau_system(dau_agents)

# Print results
for agent, efe in efe_results.items():
    print(f"EFE for {agent}: {efe}")
```

### 3 Visualize EFE Values
Plot EFE values for different agents:
```python
import matplotlib.pyplot as plt

for agent, efe in efe_results.items():
    plt.plot(efe, label=agent)

plt.xlabel("Policy Index")
plt.ylabel("Expected Free Energy (EFE)")
plt.title("EFE for Different DAU Agents")
plt.legend()
plt.show()
```

## Interpretation
- **Lower EFE values** indicate better policies (lower expected surprise).
- **Higher EFE values** suggest suboptimal or uncertain action planning.
- The balance between **instrumental** and **epistemic** components influences decision-making strategies.

## Dependencies
- Python 3+
- `pymdp`
- `numpy`
- `matplotlib`

## License
This project is open-source and available under the MIT License.

## Contact
For questions or contributions, feel free to reach out!

🚀 Happy active inference modeling! 🚀
