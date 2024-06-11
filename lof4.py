import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Load training data
train_data = pd.read_excel("C:\\code\\ps\\Qualified lots.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Load test data
test_data = pd.read_excel("C:\\code\\ps\\Subsequent lots for ongoing reliability testing.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Select features for LOF
features = ["Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"]

# Initialize dictionaries to store LOF scores
lof_train_scores = {}
lof_test_scores = {}

# Train the LOF model and compute LOF scores for each feature separately
for feature in features:
    # Prepare training and testing datasets
    X_train = train_data[[feature]]
    X_test = test_data[[feature]]
    
    # Train the LOF model for novelty detection
    lof_novelty = LocalOutlierFactor(n_neighbors=25, novelty=True)
    lof_novelty.fit(X_train)
    
    # Compute LOF scores for the training and test data
    lof_train_scores[feature] = -lof_novelty.negative_outlier_factor_
    lof_test_scores[feature] = -lof_novelty.decision_function(X_test)

# Compute the anomaly threshold using the LOF scores of the training data
all_train_scores = np.concatenate(list(lof_train_scores.values()))
mu = np.mean(all_train_scores)
sigma = np.std(all_train_scores)
z_score = 2  # Set z-score for the desired confidence level
anomaly_threshold = mu + z_score * sigma

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot LOF scores for each feature
for feature in features:
    plt.plot(test_data["Cycle"], lof_test_scores[feature], label=f'{feature} LOF', alpha=0.7)

# Plot the anomaly threshold
plt.axhline(y=anomaly_threshold, color='r', linestyle='--', label='Anomaly Threshold')

# Adding annotations
plt.text(test_data["Cycle"].max(), anomaly_threshold + 1, 'Anomaly threshold', ha='center', va='bottom', color='red', fontsize=12)

plt.xlabel('Cycle Number')
plt.ylabel('LOF')
plt.title('LOF Values Over Cycles for Test Cells')
plt.legend()
plt.grid(True)
plt.show()

# Detect anomalies based on when the LOF score crosses the threshold for each test cell
anomaly_detected = {}

for feature in features:
    anomaly_detected[feature] = None
    for i in range(len(lof_test_scores[feature])):
        if lof_test_scores[feature][i] > anomaly_threshold:
            anomaly_detected[feature] = test_data.iloc[i]['Cycle']
            break  # Stop after the first anomaly is detected

# Output anomalies detected for each test cell
for feature, cycle in anomaly_detected.items():
    print(f"Anomaly detected for {feature} at cycle number: {cycle}")
