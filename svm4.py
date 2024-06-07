import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt

# Load training and test data
training_data = pd.read_excel("C:/code/ps/Qualified lots.xlsx")
test_data = pd.read_excel("C:/code/ps/Subsequent lots for ongoing reliability testing.xlsx")

# Select features (cycle number and capacity)
features = ["Cycle"] + [f"Capacity_Cell{i}" for i in range(1, 7)]
X_train = training_data[features]
X_test = test_data[features]

# Scale features (optional, but recommended for SVM)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Shape of scaled training features:", X_train_scaled.shape)

# Define and train one-class SVM model (adjust nu for desired anomaly ratio)
model = OneClassSVM(nu=0.1, kernel="rbf")  # nu controls the fraction of outliers
model.fit(X_train_scaled)
decision_scores = model.decision_function(X_test_scaled)

# Calculate mean (mu) and standard deviation (sigma) of decision scores from training data
train_decision_scores = model.decision_function(X_train_scaled)
mu = np.mean(train_decision_scores)
sigma = np.std(train_decision_scores)

# Define anomaly threshold based on z-score
z_score = 5  # Adjust this based on your desired confidence level
threshold = mu - z_score * sigma

# Identify anomalies based on threshold
anomalies = X_test.loc[decision_scores < threshold]

# Prediction bounds based on training data
upper_bound = training_data.iloc[:, 2:].max(axis=1) + 0.03  # Maximum capacity among cells + 0.03
lower_bound = training_data.iloc[:, 2:].min(axis=1) - 0.03  # Minimum capacity among cells - 0.03

# Function to find the first anomaly cycle number for each cell
def find_first_anomaly(anomalies, test_data, cell, upper_bound, lower_bound):
    for i, row in test_data.iterrows():
        if row[f"Capacity_Cell{cell}"] > upper_bound[i] or row[f"Capacity_Cell{cell}"] < lower_bound[i]:
            first_anomaly_cycle = row['Cycle']
            print(f"First anomaly detected in Cell {cell} at Cycle {first_anomaly_cycle}")
            return first_anomaly_cycle
    print(f"No anomaly detected in Cell {cell}")
    return None

# Dictionary to store the first anomaly cycle for each cell
first_anomaly_cycles = {}

# Loop through each cell and find the first anomaly cycle
for cell in range(1, 7):
    cell_anomalies = anomalies[['Cycle', f'Capacity_Cell{cell}']]
    first_anomaly_cycles[cell] = find_first_anomaly(cell_anomalies, test_data, cell, upper_bound, lower_bound)

# Plot histogram of anomaly scores
plt.figure(figsize=(20, 20))
plt.plot(test_data["Cycle"], decision_scores, marker='.', linestyle='')
plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.xlabel('Cycle')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Scores Over Cycles')
plt.legend()
plt.grid(True)
plt.show()

# Print the first anomaly cycle for each cell
print("First anomaly cycle numbers for each cell:")
print(first_anomaly_cycles)

# Plot combined graph for all cells
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Plot training data
for cell in range(1, 7):
    plt.plot(training_data["Cycle"], training_data[f"Capacity_Cell{cell}"], color=colors[cell - 1], linestyle='-', marker='o', label=f'Training Cell {cell}')

# Plot prediction bounds based on training data
plt.plot(training_data["Cycle"], upper_bound, color='r', linestyle='--', linewidth=1, label='Prediction bound (Upper)')
plt.plot(training_data["Cycle"], lower_bound, color='r', linestyle='--', linewidth=1, label='Prediction bound (Lower)')

# Plot anomaly scores over cycles
for cell in range(1, 7):
    cell_anomalies = anomalies[['Cycle', f'Capacity_Cell{cell}']]
    plt.scatter(cell_anomalies["Cycle"], cell_anomalies[f'Capacity_Cell{cell}'], label=f'Anomalies Cell {cell}', zorder=5)

# Annotations for prediction bound and anomaly scores
plt.text(222, 0.9177, 'Prediction bound', ha='center', va='bottom', color='red')
plt.text(222, 0.88, 'Anomalies', ha='center', va='bottom', color='orange')

# Annotations for data points
plt.text(100, 0.77, 'Training samples', ha='left', va='center')

plt.xlabel('Cycle Number')
plt.ylabel('Normalized Capacity')
plt.title('Cycle Number vs Normalized Capacity for 6 Cells')
plt.legend()
plt.grid(True)
plt.show()
