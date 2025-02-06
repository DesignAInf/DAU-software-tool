import numpy as np
from pymdp.maths import softmax
from pymdp.utils import obj_array_zeros
from pymdp.agent import Agent

def compute_expected_free_energy(agent: Agent):
    """
    Compute the Expected Free Energy (EFE) for a given agent.
    """
    efe_values = []
    
    for policy in agent.policies:
        efe = 0.0
        for t in range(len(policy)):
            # Predict next state given the current state and policy
            future_states = agent.B[t] @ agent.qs
            
            # Compute expected observations under this policy
            expected_obs = agent.A @ future_states
            
            # Compute instrumental value (minimizing expected divergence from preferred observations)
            instrumental_value = np.sum(agent.C * np.log(expected_obs + 1e-8))
            
            # Compute epistemic value (reducing uncertainty)
            entropy_prior = -np.sum(expected_obs * np.log(expected_obs + 1e-8))
            entropy_posterior = -np.sum(future_states * np.log(future_states + 1e-8))
            epistemic_value = entropy_prior - entropy_posterior
            
            # EFE is the negative sum of instrumental and epistemic values
            efe += -(instrumental_value + epistemic_value)
        
        efe_values.append(efe)
    
    return efe_values

# Example usage (assuming an initialized pymdp.Agent object)
def compute_efe_for_dau_system(dau_agents):
    """
    Compute the Expected Free Energy for each agent in the DAU system.
    """
    efe_results = {}
    
    for agent_name, agent in dau_agents.items():
        efe_results[agent_name] = compute_expected_free_energy(agent)
    
    return efe_results

# Example call (assuming a dictionary of DAU agents is available)
# dau_agents = {'Designer': designer_agent, 'Artifact': artifact_agent, 'User': user_agent}
# efe_results = compute_efe_for_dau_system(dau_agents)
# print(efe_results)
