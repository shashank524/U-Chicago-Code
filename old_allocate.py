import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def calculate_risk_contributions(weights, cov_matrix):
    """
    Calculate risk contributions for each asset in the portfolio.
    """
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    asset_variances = weights * (np.dot(cov_matrix, weights))
    risk_contributions = asset_variances / portfolio_variance
    return risk_contributions

def risk_parity_allocation(returns):
    """
    Calculate the portfolio weights using Risk Parity Allocation.
    """
    cov_matrix = returns.cov()
    n_assets = len(cov_matrix)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    
    # Objective function to minimize
    def objective_function(weights):
        risk_contributions = calculate_risk_contributions(weights, cov_matrix)
        return np.sum((risk_contributions - 1 / n_assets)**2)
    
    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x}
    )
    
    # Optimization
    result = scipy.optimize.minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints)
    return result.x

def allocate_portfolio(asset_prices):
    # Calculate historical returns
    returns = asset_prices.pct_change().dropna()

    # Implement the Risk Parity Allocation strategy
    weights = risk_parity_allocation(returns)
    
    return weights

if __name__ == "__main__":
    asset_prices = pd.read_csv('./Training_Data_Case_3.csv')
    weights = allocate_portfolio(asset_prices=asset_prices)
    print(sum(weights))
    print(weights)