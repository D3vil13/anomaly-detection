import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt

# Load training data
train_data = pd.read_excel("C:\\code\\ps\\Qualified lots.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Load test data
test_data = pd.read_excel("C:\\code\\ps\\Subsequent lots for ongoing reliability testing.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Combine train and test data for anomaly detection (optional)
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Select features for anomaly detection
features = ["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"]

# Compute Mahalanobis distance for each data point
mcd = MinCovDet().fit(combined_data[features])
mahalanobis_distances = mcd.mahalanobis(combined_data[features])

# Define anomaly threshold (adjust as needed)
threshold = np.mean(mahalanobis_distances) + 3 * np.std(mahalanobis_distances)

# Visualize anomalies in test data
plt.figure(figsize=(12, 6))  # Adjust figure size as needed
plt.plot(combined_data["Cycle"], mahalanobis_distances, color='blue', label='Mahalanobis Distance')
plt.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
plt.scatter(combined_data["Cycle"][:len(train_data)], mahalanobis_distances[:len(train_data)], color='black', marker='o', s=10, label='Training Samples')

# Highlight the testing sample
test_sample_index = 157  # Replace with the actual index of your testing sample
plt.scatter(combined_data["Cycle"][test_sample_index], mahalanobis_distances[test_sample_index], color='red', marker='o', s=100, label='Testing Sample')

# Annotations
plt.annotate('Testing Sample', xy=(combined_data["Cycle"][test_sample_index], mahalanobis_distances[test_sample_index]), xytext=(200, 10), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'), fontsize=12)
plt.annotate('Training Samples', xy=(combined_data["Cycle"][len(train_data) - 1], threshold), xytext=(400, 4), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'), fontsize=12)

# Labeling and formatting
plt.xlabel('Cycle Number', fontsize=14)
plt.ylabel('Mahalanobis Distance', fontsize=14)
plt.title('Anomaly Detection using Mahalanobis Distance', fontsize=16)
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
