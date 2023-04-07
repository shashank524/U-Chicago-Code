import numpy as np
import pandas as pd
import scipy.optimize as sc

historical_asset_prices = []
data = pd.read_csv('Training Data_Case 3.csv',header=0, index_col=0)

def calculate_risk_free_rate():
    # Calculate the risk-free rate using some method or use a fixed value
    return 0.02

def getData(asset_prices):
    returns = asset_prices.pct_change().dropna()
    mean_returns = returns.mean()
    
    if len(returns) == 1:
        # If there's only one row of data, return a zero covariance matrix
        cov_matrix = pd.DataFrame(np.zeros((len(asset_prices.columns), len(asset_prices.columns))), columns=asset_prices.columns, index=asset_prices.columns)
    else:
        cov_matrix = returns.cov()
    
    return mean_returns, cov_matrix

def portfolioPerformance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def negativeSR(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_returns, p_std = portfolioPerformance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

def maxSR(mean_returns, cov_matrix, risk_free_rate=0, constraintSet=(0, 1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(num_assets))
    result = sc.minimize(negativeSR, num_assets * [1. / num_assets], args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def allocate_portfolio(asset_prices):
    global historical_asset_prices
    historical_asset_prices.append(asset_prices)
    df = pd.DataFrame(historical_asset_prices, columns=[str(i) for i in range(1, 11)])
    
    if len(df) == 1:
        # If there's only one row of data, return equal weights
        return np.repeat(1/len(asset_prices), len(asset_prices))

    risk_free_rate = calculate_risk_free_rate()
    mean_returns, cov_matrix = getData(df)
    
    # If the covariance matrix is zero, return equal weights
    if cov_matrix.eq(0).all().all():
        return np.repeat(1/len(asset_prices), len(asset_prices))

    optimized = maxSR(mean_returns, cov_matrix, risk_free_rate)
    weights = optimized.x
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
