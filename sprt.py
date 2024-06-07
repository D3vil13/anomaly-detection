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

# Initialize SPRT values and thresholds
sprt_values = np.zeros(len(test_data))
confidence_level = 0.95
z_score = norm.ppf(1 - (1 - confidence_level) / 2)
M = z_score  # System disturbance magnitude, calculated from the confidence level

# Compute mean and standard deviation for the healthy (null hypothesis) data
healthy_mean = train_data[features].mean()
healthy_std = train_data[features].std()

# Artificially introduce faulty data by adjusting the mean
faulty_mean = healthy_mean + M * healthy_std

# Calculate likelihoods for each observation in the test data
alpha = 0.05  # False positive rate
beta = 0.05   # False negative rate
A = np.log((1 - beta) / alpha)  # Threshold A
B = np.log(beta / (1 - beta))   # Threshold B

for i in range(len(test_data)):
    sprt_sum = 0
    for feature in features:
        x_i = test_data.iloc[i][feature]
        healthy_likelihood = (1 / (np.sqrt(2 * np.pi) * healthy_std[feature])) * np.exp(-0.5 * ((x_i - healthy_mean[feature]) / healthy_std[feature]) ** 2)
        faulty_likelihood = (1 / (np.sqrt(2 * np.pi) * healthy_std[feature])) * np.exp(-0.5 * ((x_i - faulty_mean[feature]) / healthy_std[feature]) ** 2)
        sprt_sum += np.log(faulty_likelihood / healthy_likelihood)
    
    sprt_values[i] = sprt_sum

# Determine anomalies based on thresholds A and B
anomalies = []
for i, value in enumerate(sprt_values):
    if value > A:
        anomalies.append(i)
        sprt_values[i] = 0  # Reset SPRT value after detecting an anomaly
    elif value < B:
        sprt_values[i] = 0  # Reset SPRT value when no fault is detected

# Visualize SPRT results
plt.figure(figsize=(10, 6))
plt.plot(test_data["Cycle"], sprt_values, marker='o', linestyle='-', label='SPRT Values')
plt.axhline(y=A, color='r', linestyle='--', label='Threshold A')
plt.axhline(y=B, color='g', linestyle='--', label='Threshold B')
plt.xlabel('Cycle')
plt.ylabel('SPRT Value')
plt.title('SPRT Values Over Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Mark anomalies on the test data
plt.scatter(test_data["Cycle"], test_data["Capacity_Cell1"], label='Normal Data')
plt.scatter(test_data.iloc[anomalies]["Cycle"], test_data.iloc[anomalies]["Capacity_Cell1"], edgecolors='r', facecolors='none', label='Anomalies')
plt.xlabel('Cycle')
plt.ylabel('Capacity_Cell1')
plt.title('Anomaly Detection using SPRT')
plt.legend()
plt.grid(True)
plt.show()
