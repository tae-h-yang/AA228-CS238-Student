import pandas as pd
import numpy as np
import os
import time

def read_csv_data(csv_filepath):
    data = pd.read_csv(csv_filepath, header=0, names=["s", "a", "r", "sp"])
    return data

def linear_to_coords(index, grid_size=10):
    """Converts a 1-based linear index to (x, y) coordinates in a grid of given size."""
    index -= 1  # Adjust for 1-based indexing
    x = index % grid_size
    y = index // grid_size
    return (x, y)

def coords_to_linear(x, y, grid_size=10):
    """Converts (x, y) coordinates to a 1-based linear index in a grid of given size."""
    return x + y * grid_size + 1  # Adjust for 1-based indexing

def write_policy_file(policy, output_filename):
    os.makedirs('./result', exist_ok=True)
    output_filepath = f'./result/{output_filename}'
    
    with open(output_filepath, 'w') as f:
        for action in policy:
            f.write(f"{action}\n")
    print(f"Policy saved to {output_filepath}")

def q_learning(input_file):
    """Estimate policy for small.csv using Batch Q-Learning."""
    start_time = time.time()

    data = read_csv_data(input_file)
    
    num_states = 100
    num_actions = 4
    discount_factor = 0.95
    learning_rate = 0.1
    num_epochs = 100  
    
    # Initialize Q-table
    Q = np.zeros((num_states, num_actions))
    
    for epoch in range(num_epochs):
        # Shuffle data each epoch for better generalization
        data = data.sample(frac=1).reset_index(drop=True)
        
        for _, row in data.iterrows():
            s = int(row["s"]) - 1 
            a = int(row["a"]) - 1  
            r = row["r"]
            sp = int(row["sp"]) - 1  
            
            Q[s, a] = Q[s, a] + learning_rate * (r + discount_factor * np.max(Q[sp]) - Q[s, a])
    
    policy = np.argmax(Q, axis=1) + 1  # Convert actions back to 1-based indexing
    
    output_filename = "small.policy"
    write_policy_file(policy, output_filename)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def value_iteration(input_file):
    """Estimate policy for small.csv using Value Iteration.""" 
    
    data = read_csv_data(input_file)
    
    num_states = 100
    num_actions = 4
    discount_factor = 0.95
    threshold = 1e-4  # Convergence threshold for value updates
    
    # Initialize V and Q-tables
    V = np.zeros(num_states)
    Q = np.zeros((num_states, num_actions))
    
    # Create dictionaries to store rewards and transition frequencies
    transitions = {}
    rewards = {}
    for _, row in data.iterrows():
        s, a, r, sp = int(row["s"]) - 1, int(row["a"]) - 1, row["r"], int(row["sp"]) - 1
        if (s, a) not in transitions:
            transitions[(s, a)] = {}
        if sp not in transitions[(s, a)]:
            transitions[(s, a)][sp] = 0
        transitions[(s, a)][sp] += 1  # Count occurrences of each (s, a) -> sp transition
        rewards[(s, a)] = r  # Same reward for identical (s, a) pairs
    
    # Value Iteration
    while True:
        delta = 0
        # Update each state's value
        for s in range(num_states):
            v = V[s]
            # Compute Q-values for all actions from state s
            for a in range(num_actions):
                if (s, a) in transitions:
                    # Compute weighted average of V[sp] based on transition frequencies
                    total_transitions = sum(transitions[(s, a)].values())
                    expected_value = sum((count / total_transitions) * V[sp] for sp, count in transitions[(s, a)].items())
                    Q[s, a] = rewards[(s, a)] + discount_factor * expected_value
                else:
                    Q[s, a] = 0  # If (s, a) not in data, assume Q(s, a) = 0
            # Update V(s) 
            V[s] = np.max(Q[s])
            delta = max(delta, abs(v - V[s]))
        if delta < threshold:
            break
    
    policy = np.argmax(Q, axis=1) + 1  # Convert actions back to 1-based indexing
    
    output_filename = "small.policy"
    write_policy_file(policy, output_filename)
    
if __name__ == "__main__":
    start_time = time.time() 
    # q_learning("./data/small.csv")
    value_iteration("./data/small.csv")

    end_time = time.time()
    print(f"Time taken for policy estimation: {end_time - start_time:.2f} seconds")
