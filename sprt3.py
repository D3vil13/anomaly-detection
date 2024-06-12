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
def calculate_sprt_values(data, A, B, M):
    sprt_values = np.zeros(len(data))
    for i in range(len(data)):
        x_i = data.iloc[i]
        # Compute likelihoods
        healthy_likelihood = norm.pdf(x_i, loc=np.mean(data), scale=np.std(data))
        faulty_likelihood = norm.pdf(x_i, loc=np.mean(data) + M * np.std(data), scale=np.std(data))
        # Update SPRT value
        if i == 0:
            sprt_values[i] = np.log(faulty_likelihood / healthy_likelihood)
        else:
            sprt_values[i] = sprt_values[i - 1] + np.log(faulty_likelihood / healthy_likelihood)
    return sprt_values

# Calculate thresholds A and B
alpha = 0.05  # False positive rate
beta = 0.05   # False negative rate
A = np.log((1 - beta) / alpha)  # Threshold A
B = np.log(beta / (1 - beta))   # Threshold B

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot SPRT values for each training cell
for idx, cell in enumerate(features, start=1):
    sprt_values_train = calculate_sprt_values(train_data[cell], A, B, M)
    plt.plot(train_data["Cycle"], sprt_values_train, label=f'Training Data Cell {idx}', color = "black")

# Plot SPRT values for each test cell
for idx, cell in enumerate(features, start=1):
    sprt_values_test = calculate_sprt_values(test_data[cell], A, B, M)
    plt.plot(test_data["Cycle"], sprt_values_test, label=f'Test Cell {idx}')

# Plot the anomaly threshold
plt.axhline(y=A, color='r', linestyle='--', label='Anomaly Threshold')

plt.xlabel('Cycle Number')
plt.ylabel('SPRT')
plt.title('SPRT Values Over Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Detect anomalies based on when the test cell curve crosses the threshold
anomaly_detected = {}
for idx, cell in enumerate(features, start=1):
    sprt_values_test = calculate_sprt_values(test_data[cell], A, B, M)
    for i in range(1, len(sprt_values_test)):
        if sprt_values_test[i] > A and sprt_values_test[i-1] <= A:
            anomaly_detected[f'Test Cell {idx}'] = test_data.iloc[i]['Cycle']
        elif sprt_values_test[i] < B and sprt_values_test[i-1] >= B:
            anomaly_detected[f'Test Cell {idx}'] = test_data.iloc[i]['Cycle']

# Output anomaly detected for each test cell
for cell, cycle in anomaly_detected.items():
    if cycle:
        print(f"Anomaly detected for {cell} at cycle number {cycle}.")
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
def calculate_sprt_values(data, A, B, M):
    sprt_values = np.zeros(len(data))
    for i in range(len(data)):
        x_i = data.iloc[i]
        # Compute likelihoods
        healthy_likelihood = norm.pdf(x_i, loc=np.mean(data), scale=np.std(data))
        faulty_likelihood = norm.pdf(x_i, loc=np.mean(data) + M * np.std(data), scale=np.std(data))
        # Update SPRT value
        if i == 0:
            sprt_values[i] = np.log(faulty_likelihood / healthy_likelihood)
        else:
            sprt_values[i] = sprt_values[i - 1] + np.log(faulty_likelihood / healthy_likelihood)
    return sprt_values

# Calculate thresholds A and B
alpha = 0.05  # False positive rate
beta = 0.05   # False negative rate
A = np.log((1 - beta) / alpha)  # Threshold A
B = np.log(beta / (1 - beta))   # Threshold B

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot SPRT values for each training cell
for idx, cell in enumerate(features, start=1):
    sprt_values_train = calculate_sprt_values(train_data[cell], A, B, M)
    plt.plot(train_data["Cycle"], sprt_values_train, label=f'Training Data Cell {idx}', color = "black")

# Plot SPRT values for each test cell
for idx, cell in enumerate(features, start=1):
    sprt_values_test = calculate_sprt_values(test_data[cell], A, B, M)
    plt.plot(test_data["Cycle"], sprt_values_test, label=f'Test Cell {idx}')

# Plot the anomaly threshold
plt.axhline(y=A, color='r', linestyle='--', label='Anomaly Threshold')

plt.xlabel('Cycle Number')
plt.ylabel('SPRT')
plt.title('SPRT Values Over Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Detect anomalies based on when the test cell curve crosses the threshold
anomaly_detected = {}
for idx, cell in enumerate(features, start=1):
    sprt_values_test = calculate_sprt_values(test_data[cell], A, B, M)
    for i in range(1, len(sprt_values_test)):
        if sprt_values_test[i] > A and sprt_values_test[i-1] <= A:
            anomaly_detected[f'Test Cell {idx}'] = test_data.iloc[i]['Cycle']
        elif sprt_values_test[i] < B and sprt_values_test[i-1] >= B:
            anomaly_detected[f'Test Cell {idx}'] = test_data.iloc[i]['Cycle']

# Output anomaly detected for each test cell
for cell, cycle in anomaly_detected.items():
    if cycle:
        print(f"Anomaly detected for {cell} at cycle number {cycle}.")
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
def calculate_sprt_values(data, A, B, M):
    sprt_values = np.zeros(len(data))
    for i in range(len(data)):
        x_i = data.iloc[i]
        # Compute likelihoods
        healthy_likelihood = norm.pdf(x_i, loc=np.mean(data), scale=np.std(data))
        faulty_likelihood = norm.pdf(x_i, loc=np.mean(data) + M * np.std(data), scale=np.std(data))
        # Update SPRT value
        if i == 0:
            sprt_values[i] = np.log(faulty_likelihood / healthy_likelihood)
        else:
            sprt_values[i] = sprt_values[i - 1] + np.log(faulty_likelihood / healthy_likelihood)
    return sprt_values

# Calculate thresholds A and B
alpha = 0.05  # False positive rate
beta = 0.05   # False negative rate
A = np.log((1 - beta) / alpha)  # Threshold A
B = np.log(beta / (1 - beta))   # Threshold B

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot SPRT values for each training cell
for idx, cell in enumerate(features, start=1):
    sprt_values_train = calculate_sprt_values(train_data[cell], A, B, M)
    plt.plot(train_data["Cycle"], sprt_values_train, label=f'Training Data Cell {idx}', color = "black")

# Plot SPRT values for each test cell
for idx, cell in enumerate(features, start=1):
    sprt_values_test = calculate_sprt_values(test_data[cell], A, B, M)
    plt.plot(test_data["Cycle"], sprt_values_test, label=f'Test Cell {idx}')

# Plot the anomaly threshold
plt.axhline(y=A, color='r', linestyle='--', label='Anomaly Threshold')

plt.xlabel('Cycle Number')
plt.ylabel('SPRT')
plt.title('SPRT Values Over Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Detect anomalies based on when the test cell curve crosses the threshold
anomaly_detected = {}
for idx, cell in enumerate(features, start=1):
    sprt_values_test = calculate_sprt_values(test_data[cell], A, B, M)
    for i in range(1, len(sprt_values_test)):
        if sprt_values_test[i] > A and sprt_values_test[i-1] <= A:
            anomaly_detected[f'Test Cell {idx}'] = test_data.iloc[i]['Cycle']
        elif sprt_values_test[i] < B and sprt_values_test[i-1] >= B:
            anomaly_detected[f'Test Cell {idx}'] = test_data.iloc[i]['Cycle']

# Output anomaly detected for each test cell
for cell, cycle in anomaly_detected.items():
    if cycle:
        print(f"Anomaly detected for {cell} at cycle number {cycle}.")
