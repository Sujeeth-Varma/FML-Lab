import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.eye(m)  # identity matrix of shape (m, m)
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(-np.sum(np.square(diff)) / (2 * k**2))  # Gaussian kernel
    return weights

def localWeight(point, xmat, ymat, k, lambda_reg=1e-5):
    wei = kernel(point, xmat, k)
    
    # Add regularization term to avoid singular matrix
    A = xmat.T @ (wei @ xmat) + lambda_reg * np.eye(xmat.shape[1])
    
    # Compute W with regularization
    W = np.linalg.inv(A) @ (xmat.T @ (wei @ ymat.T))
    return W

def localWeightRegression(xmat, ymat, k, lambda_reg=1e-5):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] @ localWeight(xmat[i], xmat, ymat, k, lambda_reg)
    return ypred

# Load data points
data = pd.read_csv(r"./datasets/10.csv")
bill = np.array(data['total_bill'])
tip = np.array(data['tip']) 

# Prepare and add 1 in bill for the intercept term
m = len(bill)
X = np.vstack((np.ones(m), bill)).T  # Shape X as (m, 2)

# Set k here
k = 0.2
ypred = localWeightRegression(X, tip, k)

# Sort X for plotting
SortIndex = X[:, 1].argsort()
xsort = X[SortIndex][:, 1]

# Plot the data points and the regression line
fig, ax = plt.subplots()
ax.scatter(bill, tip, color='green', label='Data points')
ax.plot(xsort, ypred[SortIndex], color='red', linewidth=5, label='LWR Fit')
ax.set_xlabel('Total bill')
ax.set_ylabel('Tip')
ax.legend()
plt.show()
