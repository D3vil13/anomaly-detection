import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load training data
train_data = pd.read_excel("C:\\code\\ps\\Qualified lots.xlsx")

# Load test data
test_data = pd.read_excel("C:\\code\\ps\\Subsequent lots for ongoing reliability testing.xlsx")

# Define a function to calculate Mahalanobis Distance
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    md = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return md

# Extract cycle numbers and capacity data
train_cycles = train_data.iloc[:, 0].values
train_capacities = train_data.iloc[:, 1:].values

test_cycles = test_data.iloc[:, 0].values
test_capacities = test_data.iloc[:, 1:].values

# Calculate mean vector and covariance matrix for training data
mean_vector = np.mean(train_capacities, axis=0)
cov_matrix = np.cov(train_capacities, rowvar=False)

# Calculate MD scores for training data
train_md_scores = np.array([mahalanobis_distance(x, mean_vector, cov_matrix) for x in train_capacities])

# Calculate the anomaly threshold using the mean and standard deviation of the training MD scores
threshold = np.mean(train_md_scores) + 5 * np.std(train_md_scores)

# Calculate MD scores for test data
test_md_scores = np.array([[mahalanobis_distance(x, mean_vector, cov_matrix) for x in test_capacities[:,i]] for i in range(test_capacities.shape[1])])

# Plot the MD scores and anomaly threshold
plt.figure(figsize=(12, 8))

# Plot MD scores for training data
plt.plot(train_cycles, train_md_scores, label='Training Data', color='black')

# Plot MD scores for test data
for i in range(test_capacities.shape[1]):
    plt.plot(test_cycles, test_md_scores[i], label=f'Test Cell {i+1}')

# Plot the anomaly threshold
plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')

plt.xlabel('Cycle Number')
plt.ylabel('Mahalanobis Distance')
plt.title('Mahalanobis Distance Scores for Training and Test Data')
plt.legend()
plt.grid(True)
plt.show()

# Print the cycle number at which each test cell crosses the threshold
for i in range(test_capacities.shape[1]):
    anomaly_cycle = test_cycles[np.where(test_md_scores[i] > threshold)[0][0]]
    print(f"Test Cell {i+1} crosses the anomaly threshold at cycle number {anomaly_cycle}.")
