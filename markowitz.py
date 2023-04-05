import numpy as np
import pandas as pd
import scipy.optimize as opt

def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def markowitz_optimization(returns_df):
    n_assets = len(returns_df.columns)
    cov_matrix = returns_df.cov()
    
    initial_weights = np.repeat(1 / n_assets, n_assets)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = opt.minimize(calculate_portfolio_variance, initial_weights, args=(cov_matrix,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def allocate_portfolio(asset_prices):
    returns_df = asset_prices.pct_change().dropna()
    weights = markowitz_optimization(returns_df)
    return weights

if __name__ == "__main__":
    asset_prices = pd.read_csv('./Training_Data_Case_3.csv')
    weights = allocate_portfolio(asset_prices=asset_prices)
    print(sum(weights))
    print(weights)