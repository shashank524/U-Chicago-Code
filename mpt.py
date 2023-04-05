import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load historical data for the 10 stocks
data = pd.read_csv('stock_data.csv', index_col='date')

# Calculate returns and expected returns for each stock
returns = data.pct_change().dropna()
mu = returns.mean()

# Calculate the covariance matrix and correlation matrix
cov = returns.cov()
corr = returns.corr()

# Define the objective function to minimize (negative Sharpe ratio)
def neg_sharpe(weights, mu, cov):
    port_return = np.dot(weights, mu)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return -port_return / port_std

# Define constraints (weights sum to 1, no short selling)
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x})

# Define initial guess for weights
x0 = np.ones(len(mu)) / len(mu)

# Minimize the negative Sharpe ratio to find optimal weights
result = minimize(neg_sharpe, x0, args=(mu, cov), method='SLSQP', constraints=cons)

# Calculate optimal portfolio statistics
optimal_weights = result.x
optimal_return = np.dot(optimal_weights, mu)
optimal_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov, optimal_weights)))
optimal_sharpe = -result.fun

# Print results
print('Optimal weights:', optimal_weights)
print('Optimal return:', optimal_return)
print('Optimal standard deviation:', optimal_std)
print('Optimal Sharpe ratio:', optimal_sharpe)
