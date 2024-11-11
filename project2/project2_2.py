import sys
import pandas as pd
import numpy as np
import os

def read_csv_data(csv_filepath):
    data = pd.read_csv(csv_filepath, header=0, names=["s", "a", "r", "sp"])
    return data

def linear_to_coords(index, grid_size=10):
    """Converts a 1-based linear index to (x, y) coordinates in a 1-based column-major order grid of given size."""
    index -= 1  # Adjust for 1-based indexing
    x = (index % grid_size) + 1  # Row number, 1-based
    y = (index // grid_size) + 1 # Column number, 1-based
    return (x, y)

def coords_to_linear(x, y, grid_size=10):
    """Converts (x, y) coordinates to a 1-based linear index in a column-major order grid of given size."""
    return (y - 1) * grid_size + x

def build_transition_rewards(data, num_states=100, num_actions=4):
    """Constructs transition and reward tables from the dataset."""
    transitions = {state: {action: [] for action in range(num_actions)} for state in range(1, num_states + 1)}
    rewards = np.zeros((num_states + 1, num_actions))  # Initialize rewards for 1-based indexing
    
    for _, row in data.iterrows():
        state = int(row['s'])
        action = int(row['a']) - 1  # Convert action to 0-based index for internal use
        reward = float(row['r'])
        next_state = int(row['sp'])
        
        transitions[state][action].append(next_state)
        rewards[state, action] = reward
    
    return transitions, rewards

def value_iteration_with_interpolation(transitions, rewards, num_states=100, num_actions=4, discount_factor=0.95, theta=1e-6, grid_size=10):
    """Performs Value Iteration with interpolation for states with no actions or rewards."""
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    def interpolate_value(state):
        x, y = linear_to_coords(state, grid_size)
        neighbors = []
        # Use the action mappings for neighbor directions
        if x > 1:  # Left (x - 1, y)
            neighbors.append(V[coords_to_linear(x - 1, y, grid_size) - 1])
        if x < grid_size:  # Right (x + 1, y)
            neighbors.append(V[coords_to_linear(x + 1, y, grid_size) - 1])
        if y < grid_size:  # Up (x, y + 1)
            neighbors.append(V[coords_to_linear(x, y + 1, grid_size) - 1])
        if y > 1:  # Down (x, y - 1)
            neighbors.append(V[coords_to_linear(x, y - 1, grid_size) - 1])
        return np.mean(neighbors) if neighbors else 0

    while True:
        delta = 0
        for state in range(1, num_states + 1):
            action_values = []
            for action in range(num_actions):
                next_states = transitions[state][action]
                
                if next_states:
                    expected_value = np.mean([V[sp - 1] for sp in next_states])  # Adjust for 1-based index
                    action_value = rewards[state - 1, action] + discount_factor * expected_value
                else:
                    action_value = interpolate_value(state)
                
                action_values.append(action_value)
                
            best_action_value = max(action_values)
            delta = max(delta, abs(best_action_value - V[state - 1]))
            V[state - 1] = best_action_value
            policy[state - 1] = np.argmax(action_values) + 1  # Convert to 1-based actions
        
        if delta < theta:
            break
    
    return policy  # Return policy directly aligned with 1-based states

def write_policy_file(policy, output_filename):
    """Writes the policy to a file in the ./result/ directory."""
    os.makedirs('./result', exist_ok=True)
    output_filepath = f'./result/{output_filename}'
    
    with open(output_filepath, 'w') as f:
        for action in policy:
            f.write(f"{action}\n")
    print(f"Policy saved to {output_filepath}")

def estimate_small(input_file):
    """Estimate policy for small.csv using Value Iteration."""
    data = read_csv_data(input_file)
    
    num_states = 100
    num_actions = 4
    discount_factor = 0.95

    transitions, rewards = build_transition_rewards(data, num_states, num_actions)

    # Calculate optimal policy using Value Iteration
    policy = value_iteration_with_interpolation(transitions, rewards, num_states, num_actions, discount_factor)

    output_filename = f"{os.path.splitext(os.path.basename(input_file))[0]}.policy"
    write_policy_file(policy, output_filename)

def main():
    if len(sys.argv) != 2:
        print("Usage: python project2.py <csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    if os.path.basename(input_file) == "small.csv":
        estimate_small(f"./data/{input_file}")
    else:
        print("This script currently only supports 'small.csv'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
