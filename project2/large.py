import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict

NUM_STATES = 302020
NUM_ACTIONS = 9
DISCOUNT_FACTOR = 0.95
THRESHOLD = 1e-4  

def read_csv_data(csv_filepath):
    data = pd.read_csv(csv_filepath, header=0, names=["s", "a", "r", "sp"])
    return data

def write_policy_file(policy, output_filename="large.policy"):
    os.makedirs('./result', exist_ok=True)
    output_filepath = f'./result/{output_filename}'
    with open(output_filepath, 'w') as f:
        for action in policy:
            f.write(f"{action}\n")
    print(f"Policy saved to {output_filepath}")

def value_iteration_large(input_file):
    """Estimate policy for large.csv using Value Iteration with proper indexing."""
    data = read_csv_data(input_file)
    
    V = np.zeros(NUM_STATES + 1)  # +1 to account for 1-based indexing of states
    Q = np.zeros((NUM_STATES + 1, NUM_ACTIONS + 1))  # +1 for 1-based indexing

    transitions = defaultdict(lambda: defaultdict(int))
    rewards = {}

    for _, row in data.iterrows():
        s, a, r, sp = int(row["s"]), int(row["a"]), row["r"], int(row["sp"])
        transitions[(s, a)][sp] += 1
        rewards[(s, a)] = r

    while True:
        delta = 0
        for s in range(1, NUM_STATES + 1):  
            v = V[s]
            for a in range(1, NUM_ACTIONS + 1):  # Actions 1 to 9
                if (s, a) in transitions:
                    total_transitions = sum(transitions[(s, a)].values())
                    expected_value = sum((count / total_transitions) * V[sp] for sp, count in transitions[(s, a)].items())
                    Q[s, a] = rewards[(s, a)] + DISCOUNT_FACTOR * expected_value
                else:
                    Q[s, a] = 0  # Assume Q(s, a) = 0 if (s, a) not in data
            V[s] = np.max(Q[s, 1:])  
            delta = max(delta, abs(v - V[s]))
        
        if delta < THRESHOLD:
            break
    
    policy = np.zeros(NUM_STATES, dtype=int)
    for s in range(1, NUM_STATES + 1):
        policy[s - 1] = np.argmax(Q[s, 1:]) + 1  # Store optimal action for each state (1-based action)

    write_policy_file(policy, "large.policy")

if __name__ == "__main__":
    start_time = time.time()
    value_iteration_large("./data/large.csv")
    print(f"Time taken for policy estimation: {time.time() - start_time:.2f} seconds")
