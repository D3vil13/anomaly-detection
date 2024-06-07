import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Load training data
train_data = pd.read_excel("C:\\code\\ps\\Qualified lots.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Load test data
test_data = pd.read_excel("C:\\code\\ps\\Subsequent lots for ongoing reliability testing.xlsx", usecols=["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"])

# Select features for anomaly detection
features = ["Cycle", "Capacity_Cell1", "Capacity_Cell2", "Capacity_Cell3", "Capacity_Cell4", "Capacity_Cell5", "Capacity_Cell6"]

# Detect anomalies using LOF on combined data
lof_detector = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
anomaly_scores = lof_detector.fit_predict(train_data[features])

# Convert scores to positive values
anomaly_scores = -anomaly_scores

# Define anomaly threshold (adjust as needed)
threshold = anomaly_scores.mean() + 3 * anomaly_scores.std()

# Visualize anomalies in test data
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.plot(train_data["Cycle"], train_data["Capacity_Cell1"], 'k.', label="Training samples")
plt.plot(test_data["Cycle"], test_data["Capacity_Cell1"], 'b-', label="Testing sample")

# Highlight anomaly threshold
plt.axhline(y=threshold, color='r', linestyle='--', label="Anomaly threshold")

# Highlight a specific anomaly point
anomaly_index = 170  # Change this to the index of the anomaly you want to highlight
plt.plot(test_data["Cycle"][anomaly_index], test_data["Capacity_Cell1"][anomaly_index], 'ro', markersize=8)
plt.text(test_data["Cycle"][anomaly_index] - 10, test_data["Capacity_Cell1"][anomaly_index] + 0.5,
         f"X {test_data['Cycle'][anomaly_index]}\nY {test_data['Capacity_Cell1'][anomaly_index]:.3f}", 
         fontsize=10, bbox={'facecolor': 'lightblue', 'alpha': 0.5})

plt.xlabel("Cycle Number")
plt.ylabel("LOF")
plt.title("Anomaly Detection using LOF")
plt.legend()
plt.grid(True)
plt.show()