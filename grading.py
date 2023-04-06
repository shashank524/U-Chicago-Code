import pandas as pd
import numpy as np
import scipy

def calculate_risk_contributions(weights, cov_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    asset_variances = weights * (np.dot(cov_matrix, weights))
    risk_contributions = asset_variances / portfolio_variance
    return risk_contributions

def risk_parity_allocation(returns):
    cov_matrix = returns.cov()
    n_assets = len(cov_matrix)
    initial_weights = np.repeat(1 / n_assets, n_assets)

    def objective_function(weights):
        risk_contributions = calculate_risk_contributions(weights, cov_matrix)
        return np.sum((risk_contributions - 1 / n_assets)**2)

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x}
    )

    result = scipy.optimize.minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints)
    return result.x

def allocate_portfolio(asset_prices):
    global historical_prices
    historical_prices = historical_prices.append(pd.Series(asset_prices, index=historical_prices.columns), ignore_index=True)
    returns = historical_prices.pct_change().dropna()
    weights = risk_parity_allocation(returns)
    return weights

def grading(testing):
    weights = np.full(shape=(len(testing.index), 10), fill_value=0.0)
    for i in range(0, len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i, :])))
        positive = np.absolute(unnormed)
        normed = positive / np.sum(positive)
        weights[i] = list(normed)
    capital = [1]
    for i in range(len(testing) - 1):
        shares = capital[-1] * np.array(weights[i]) / np.array(testing.iloc[i, :])
        capital.append(float(np.matmul(np.reshape(shares, (1, 10)), np.array(testing.iloc[i+1, :]))))
    returns = (np.array(capital[1:]) - np.array(capital[:-1])) / np.array(capital[:-1])
    return np.mean(returns) / np.std(returns) * (252 ** 0.5), capital, weights

# Initialize the historical_prices DataFrame with column names
historical_prices = pd.DataFrame(columns=[f'Asset_{i}' for i in range(10)])

data = pd.read_csv('./Training_Data_Case_3.csv',header=0, index_col=0)

print(grading(data))
