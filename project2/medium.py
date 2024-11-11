# import pandas as pd
# import numpy as np
# from scipy.spatial import distance
# import time

# # Load the dataset
# dataset = pd.read_csv("data/medium.csv")

# # Set the size of state space and action space
# 𝖲 = 50000
# 𝖠 = 7

# # Define reachable states of s
# def reachableStates(s):
#     return [s + i + 500 * j for i in range(-15, 16) for j in range(-3, 4) if 1 <= s + i + 500 * j <= 50000]

# # Infer transition and reward function
# def inferTransitionAndReward(dataset, 𝖲, 𝖠):
#     N = {}
#     Np = {}
#     ρ = {}

#     for _, row in dataset.iterrows():
#         s, a, r, sp = int(row['s']), int(row['a']), row['r'], int(row['sp'])
#         ρ[(s, a)] = ρ.get((s, a), 0) + r
#         N[(s, a)] = N.get((s, a), 0) + 1
#         Np[(s, a, sp)] = Np.get((s, a, sp), 0) + 1

#     T, R = {}, {}
#     for key in Np.keys():
#         T[key] = Np[key] / N[(key[0], key[1])]
#     for key in ρ.keys():
#         R[key] = ρ[key] / N[key]

#     return T, R

# # Value iteration with Gauss-Seidel update
# def valueIterationGaussSeidel(𝖲, 𝖠, dataset, reachableStates, γ, ϵ, reachableStateSpace=range(1, 𝖲 + 1)):
#     T, R = inferTransitionAndReward(dataset, 𝖲, 𝖠)
#     δ = ϵ * (1 - γ) / γ
#     bellmanResidual = δ + 1
#     U = np.zeros(𝖲)
#     Up = np.zeros(𝖲)
#     π = np.zeros(𝖲, dtype=int)

#     immediateReward = np.zeros((𝖲, 𝖠))
#     for s in range(𝖲):
#         for a in range(𝖠):
#             immediateReward[s, a] = R.get((s + 1, a + 1), 0)

#     k = 1
#     while bellmanResidual > δ:
#         for s in reachableStateSpace:
#             sumOfDiscountedFutureRewards = np.zeros(𝖠)
#             for a in range(𝖠):
#                 sumOfDiscountedFutureRewards[a] = γ * sum(
#                     T.get((s, a + 1, sp), 0) * Up[sp - 1] for sp in reachableStates(s)
#                 )
#             Up[s - 1], π[s - 1] = max(
#                 [(immediateReward[s - 1, a] + sumOfDiscountedFutureRewards[a], a + 1) for a in range(𝖠)]
#             )

#         bellmanResidual = np.max(np.abs(Up - U))
#         k += 1
#         U = Up.copy()

#     return Up, π

# # Solution parameters
# γ = 0.99
# ϵ = 1000.0

# # Solve and record the timing
# start_time = time.time()
# U, π = valueIterationGaussSeidel(𝖲, 𝖠, dataset, reachableStates, γ, ϵ)
# end_time = time.time()
# t = end_time - start_time
# print(f"Execution time: {t:.2f} seconds")

# # Extract policy from value function
# def write_policy_file(policy, output_filename="medium.policy"):
#     """Writes the policy to a file in the ./result/ directory."""
#     import os
#     os.makedirs('./result', exist_ok=True)
#     with open(f'./result/{output_filename}', 'w') as f:
#         for action in policy:
#             f.write(f"{action}\n")
#     print(f"Policy saved to ./result/{output_filename}")

# # Save the policy
# write_policy_file(π)

import pandas as pd
import numpy as np
from scipy.spatial import distance
import time

# Define reachable states of s
def reachable_states(state):
    return [state + i + 500 * j for i in range(-15, 16) for j in range(-3, 4) if 1 <= state + i + 500 * j <= num_states]

# Infer transition and reward function
def infer_transition_and_reward(dataset, num_states, num_actions):
    N = {}
    Np = {}
    rewards_sum = {}

    for _, row in dataset.iterrows():
        s, a, r, sp = int(row['s']), int(row['a']), row['r'], int(row['sp'])
        rewards_sum[(s, a)] = rewards_sum.get((s, a), 0) + r
        N[(s, a)] = N.get((s, a), 0) + 1
        Np[(s, a, sp)] = Np.get((s, a, sp), 0) + 1

    transition_prob, reward = {}, {}
    for key in Np.keys():
        transition_prob[key] = Np[key] / N[(key[0], key[1])]
    for key in rewards_sum.keys():
        reward[key] = rewards_sum[key] / N[key]

    return transition_prob, reward

# Value iteration with Gauss-Seidel update
def value_iteration_gauss_seidel(num_states, num_actions, dataset, reachable_states, gamma, epsilon, reachable_state_space=range(1, num_states + 1)):
    transition_prob, reward = infer_transition_and_reward(dataset, num_states, num_actions)
    delta_threshold = epsilon * (1 - gamma) / gamma
    bellman_residual = delta_threshold + 1
    U = np.zeros(num_states)
    U_new = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    immediate_reward = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            immediate_reward[s, a] = reward.get((s + 1, a + 1), 0)

    while bellman_residual > delta_threshold:
        for s in reachable_state_space:
            sum_of_discounted_future_rewards = np.zeros(num_actions)
            for a in range(num_actions):
                sum_of_discounted_future_rewards[a] = gamma * sum(
                    transition_prob.get((s, a + 1, sp), 0) * U_new[sp - 1] for sp in reachable_states(s)
                )
            U_new[s - 1], policy[s - 1] = max(
                [(immediate_reward[s - 1, a] + sum_of_discounted_future_rewards[a], a + 1) for a in range(num_actions)]
            )

        bellman_residual = np.max(np.abs(U_new - U))
        U = U_new.copy()

    return U_new, policy

def write_policy_file(policy, output_filename="medium.policy"):
    """Writes the policy to a file in the ./result/ directory."""
    import os
    os.makedirs('./result', exist_ok=True)
    with open(f'./result/{output_filename}', 'w') as f:
        for action in policy:
            f.write(f"{action}\n")
    print(f"Policy saved to ./result/{output_filename}")

if __name__ == "__main__":
    dataset = pd.read_csv("data/medium.csv")

    num_states = 50000
    num_actions = 7

    gamma = 0.99
    epsilon = 1000.0

    start_time = time.time()
    U, policy = value_iteration_gauss_seidel(num_states, num_actions, dataset, reachable_states, gamma, epsilon)
    end_time = time.time()
    t = end_time - start_time
    print(f"Execution time: {t:.2f} seconds")

    write_policy_file(policy)
