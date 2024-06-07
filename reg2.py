import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t, f
import matplotlib.pyplot as plt

# Define the two-stage power law model
def two_stage_power_law(N, K1, b1, K2, b2):
    return K1 * (N ** b1) + K2 * (N ** b2)

# Load training and test data from Excel files (adjust paths)
training_data = pd.read_excel("C:/code/ps/Qualified lots.xlsx")
test_data = pd.read_excel("C:/code/ps/Subsequent lots for ongoing reliability testing.xlsx")

# List to store results for each cell
results = []

# Fit the model for each cell's data
for cell in range(1, 7):
    # Extract cycle number and normalized discharge capacity for training and testing
    N_train = training_data['Cycle'].values
    NDC_train = training_data[f'Capacity_Cell{cell}'].values
    N_test = test_data['Cycle'].values
    NDC_test = test_data[f'Capacity_Cell{cell}'].values

    # Fit the model to the training data
    popt, pcov = curve_fit(two_stage_power_law, N_train, NDC_train, p0=[0.01, 0.5, 0.001, 1.5])

    # Calculate confidence intervals for model coefficients
    alpha = 0.05  # 95% confidence interval
    dof = max(0, len(N_train) - len(popt))
    t_val = t.ppf(1.0 - alpha / 2., dof)
    sigma = np.sqrt(np.diag(pcov))

    conf_intervals = np.array([popt - t_val * sigma, popt + t_val * sigma]).T

    def prediction_bounds(N, popt, pcov, confidence_level=0.95):
        # Calculate predictions
        y_pred = two_stage_power_law(N, *popt)

        # Calculate mean squared error
        residuals = NDC_train - two_stage_power_law(N_train, *popt)
        mse = np.mean(residuals**2)

        # Calculate F-statistic for the given confidence level
        d = len(popt)
        n = len(N_train)
        f_val = f.ppf(confidence_level, d, n - d)

        # Calculate the Jacobian matrix
        J = np.zeros((n, d))
        delta = 1e-6
        for i in range(d):
            popt_delta = popt.copy()
            popt_delta[i] += delta
            J[:, i] = (two_stage_power_law(N_train, *popt_delta) - two_stage_power_law(N_train, *popt)) / delta

        # Calculate the variance-covariance matrix
        cov_matrix = np.dot(J.T, J) * mse / (n - d)

        # Ensure the covariance matrix is positive definite
        if np.any(np.diag(cov_matrix) < 0):
            print("Warning: Negative values found in cov_matrix diagonal")
            # Consider alternative confidence interval methods here

        # Add a small value (e.g., 1e-8) to the diagonal for better conditioning
        cov_matrix += np.eye(len(cov_matrix)) * 1e-8

        # Calculate the prediction bounds
        pred_bound = f_val * np.sqrt(np.diag(np.dot(np.dot(J, np.linalg.inv(cov_matrix)), J.T)))
        pred_bound = np.resize(pred_bound, y_pred.shape)

        return y_pred, y_pred - pred_bound, y_pred + pred_bound

    # Compute prediction bounds for test data
    y_pred_train, lower_bound_train, upper_bound_train = prediction_bounds(N_train, popt, pcov)
    y_pred_test, lower_bound_test, upper_bound_test = prediction_bounds(N_test, popt, pcov)

    # Detect anomalies in test data
    anomalies = np.where((NDC_test < lower_bound_test) | (NDC_test > upper_bound_test))

    # Store results
    results.append({
        'cell': cell,
        'popt': popt,
        'pcov': pcov,
        'conf_intervals': conf_intervals,
        'y_pred_train': y_pred_train,
        'lower_bound_train': lower_bound_train,
        'upper_bound_train': upper_bound_train,
        'y_pred_test': y_pred_test,
        'lower_bound_test': lower_bound_test,
        'upper_bound_test': upper_bound_test,
        'anomalies': anomalies
    })

# Plot combined graph for all cells
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y']

for cell in range(1, 7):
    N_train = training_data['Cycle'].values
    NDC_train = training_data[f'Capacity_Cell{cell}'].values
    N_test = test_data['Cycle'].values
    NDC_test = test_data[f'Capacity_Cell{cell}'].values
    
    plt.plot(N_train, NDC_train, color=colors[cell-1], linestyle='-', marker='o', label=f'Training Data Cell {cell}')
    plt.plot(N_test, NDC_test, color=colors[cell-1], linestyle='--', marker='x', label=f'Test Data Cell {cell}')

plt.xlabel('Cycle Number')
plt.ylabel('Normalized Discharge Capacity')
plt.title('Cycle Number vs Normalized Capacity for 6 Cells')
plt.legend()
plt.show()
