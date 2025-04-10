import numpy as np
import pandas as pd

# Define the states
states = ["Facebook Ad", "Website Visit", "Email Click", "Conversion"]

# Transition matrix:
# Each row is a current state and each column is a next state.
P = np.array([
    [0.5,  0.15, 0.1,  0.25],  # From Facebook Ad
    [0.0,  0.6,  0.2,  0.2],   # From Website Visit
    [0.0,  0.0,  0.7,  0.3],   # From Email Click
    [0.0,  0.0,  0.0,  1.0]    # From Conversion (absorbing state)
])

# Display the transition matrix in a friendly format
df_P = pd.DataFrame(P, index=states, columns=states)
print("Transition Matrix:")
print(df_P)

# Identify the indices: "Conversion" is our absorbing state (index 3), others are transient.
absorbing_states = [3]  # Conversion
transient_states = [0, 1, 2]  # Facebook Ad, Website Visit, Email Click

# Extract the Q matrix (transitions among transient states) and the R matrix (transitions to the absorbing state)
Q = P[np.ix_(transient_states, transient_states)]
R = P[np.ix_(transient_states, absorbing_states)]

# Compute the Fundamental Matrix: N = (I - Q)^(-1)
I = np.eye(len(Q))       # Identity matrix matching Q's dimensions
N = np.linalg.inv(I - Q)

# Compute absorption probabilities: B = N * R
B = np.dot(N, R)
conversion_probs = B.flatten()  # Flatten to get conversion probabilities from each transient state

# Map each transient state to its conversion probability
conversion_dict = {states[i]: conversion_probs[idx] for idx, i in enumerate(transient_states)}

print("\nConversion Probabilities from Each Transient State:")
for state, prob in conversion_dict.items():
    print(f"{state}: {prob:.3f}")

# Assume an initial customer distribution: 40% at Facebook Ad, 40% at Website Visit, 20% at Email Click.
initial_distribution = np.array([0.4, 0.4, 0.2])
overall_conversion_prob = np.dot(initial_distribution, conversion_probs)
print(f"\nOverall Conversion Probability: {overall_conversion_prob:.3f}")


def compute_overall_conversion(P_modified, initial_distribution, transient_states, absorbing_state_idx):
    """
    Compute the overall conversion probability given a modified transition matrix.
    """
    Q_mod = P_modified[np.ix_(transient_states, transient_states)]
    R_mod = P_modified[np.ix_(transient_states, [absorbing_state_idx])]
    I_mod = np.eye(len(Q_mod))
    N_mod = np.linalg.inv(I_mod - Q_mod)
    B_mod = np.dot(N_mod, R_mod)
    conversion_probs_mod = B_mod.flatten()
    return np.dot(initial_distribution, conversion_probs_mod)

# Remove the "Website Visit" (state index 1) by setting all its outgoing probabilities to 0.
P_removed = P.copy()
P_removed[1, :] = 0

# Since "Website Visit" no longer contributes, adjust the transient states and initial distribution.
new_transient_states = [0, 2]  # Now only "Facebook Ad" and "Email Click"
new_initial_distribution = np.array([0.4, 0.2])  # Reassign the initial probabilities accordingly

# Compute the overall conversion probability with the modified transition matrix.
overall_conversion_removed = compute_overall_conversion(P_removed, new_initial_distribution, new_transient_states, absorbing_state_idx=3)

print(f"\nOverall Conversion Probability after removing 'Website Visit': {overall_conversion_removed:.3f}")

# The removal effect is the drop in overall conversion probability:
removal_effect = overall_conversion_prob - overall_conversion_removed
print(f"Removal Effect for 'Website Visit': {removal_effect:.3f}")