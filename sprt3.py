import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load training data
train_data = pd.read_excel("C:\\code\\ps\\Qualified lots.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Load test data
test_data = pd.read_excel("C:\\code\\ps\\Subsequent lots for ongoing reliability testing.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Select features for SPRT
features = ["Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"]

# Initialize SPRT thresholds
confidence_level = 0.95
z_score = norm.ppf(1 - (1 - confidence_level) / 2)
M = z_score  # System disturbance magnitude, calculated from the confidence level

# Function to calculate SPRT values for each cell
def calculate_sprt_values(test_data, cell, A, B, M):
    sprt_values = []
    sprt_sum = 0
    for i in range(len(test_data)):
        x_i = test_data.iloc[i][cell]
        # Compute likelihoods
        healthy_likelihood = norm.pdf(x_i, loc=train_data[cell].mean(), scale=train_data[cell].std())
        faulty_likelihood = norm.pdf(x_i, loc=train_data[cell].mean() + M * train_data[cell].std(), scale=train_data[cell].std())
        # Update SPRT sum
        sprt_sum += np.log(faulty_likelihood / healthy_likelihood)
        sprt_values.append(sprt_sum)
    return sprt_values

# Initialize a dictionary to store SPRT values for each cell
sprt_values_dict = {}

# Calculate thresholds A and B
alpha = 0.05  # False positive rate
beta = 0.05   # False negative rate
A = np.log((1 - beta) / alpha)  # Threshold A
B = np.log(beta / (1 - beta))   # Threshold B

# Loop through each cell and calculate SPRT values
for cell in features:
    sprt_values = calculate_sprt_values(test_data, cell, A, B, M)
    sprt_values_dict[cell] = sprt_values

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot SPRT values for each cell
for cell in features:
    plt.plot(test_data["Cycle"], sprt_values_dict[cell], label=f'SPRT Values {cell}')

# Plot training values as black lines
for cell in features:
    plt.plot(train_data["Cycle"], [0] * len(train_data), color='black', linewidth=1)

# Plot the anomaly threshold
plt.axhline(y=A, color='r', linestyle='--', label='Anomaly Threshold')

# Adding annotations
plt.text(400, A + 1, 'Anomaly threshold', ha='center', va='bottom', color='red', fontsize=12)
plt.text(100, -10, 'Training samples', ha='center', va='bottom', color='black', fontsize=12)

plt.xlabel('Cycle Number')
plt.ylabel('SPRT')
plt.title('SPRT Values Over Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Detect anomalies based on when the test cell curve crosses the threshold
anomaly_detected = {}
for cell in features:
    sprt_values = sprt_values_dict[cell]
    anomaly_detected[cell] = None
    for i in range(1, len(sprt_values)):
        if sprt_values[i] > A and sprt_values[i-1] <= A:
            anomaly_detected[cell] = test_data.iloc[i]['Cycle']
        elif sprt_values[i] < B and sprt_values[i-1] >= B:
            anomaly_detected[cell] = test_data.iloc[i]['Cycle']

# Output anomaly detected for each test cell
for cell, cycle in anomaly_detected.items():
    if cycle:
        print(f"Anomaly detected for {cell} at cycle number {cycle}.")
