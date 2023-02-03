import datetime as dt
import logging
import os
from typing import Mapping, Optional

import algotrading
import matplotlib.pyplot as plt
import mplfinance
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def ma_discount(df, param={"MA": 50}):
    """
    The discount rate using moving average as baseline.
    """
    MA = param["MA"]
    result = (
        -(df["Adj Close"] - df["Adj Close"].rolling(MA).mean()) / df["Adj Close"] * 100
    )
    return result


def linear_regression_gains(df, param={"MA": 90}):
    """
    The potential gain rate after 1 year use linear model
    """

    def momentum(closes):
        returns = np.log(closes)
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        # annualize slope and multiply by R^2
        return ((1 + slope) ** 252) * (rvalue**2)

    MA = param["MA"]
    result = df.rolling(MA)["Adj Close"].apply(momentum, raw=False) - 1
    return result


def calc_volatility(stock_name_list, output_file_name=None):
    """
    Calculates the volatility of a set of stock prices and plots the results as a bar graph.
    https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
    """
    stocks = [algotrading.stock.Stock(name) for name in stock_name_list]
    for stock in stocks:
        stock.read_data()
    data = {stock.name: stock.df["Close"] for stock in stocks}
    data = pd.DataFrame(data)
    returns = data.pct_change().apply(lambda x: np.log(1 + x))
    volatility = returns.std().apply(lambda x: x * np.sqrt(250))
    volatility.sort_values(ascending=True, inplace=True)
    average = volatility.mean().round(2)

    fig, ax = plt.subplots()
    volatility.plot(kind="bar", ax=ax)
    ax.axhline(y=average, color="y", linestyle="--", label="volatility")
    ax.set_title("volatility index")
    ax.legend()

    if output_file_name:
        fig.savefig(output_file_name, dpi=300)
        plt.close(fig)
        return output_file_name
    else:
        return ax


def generate_portfolio(stock_name_list, result_dir=None):
    """
    https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
    """
    test = {}
    for stock_name in stock_name_list:
        stock = algotrading.stock.Stock(stock_name)
        end = end = dt.datetime.now()
        start = start = end - dt.timedelta(3000)
        stock.read_data(start, end)
        test[stock_name] = stock.df["Close"]
    test = pd.DataFrame(test)
    markdown_notes = "portfolio analysis \n\n\n\n"

    # Yearly returns for individual companies
    ind_er = test.resample("Y").last().pct_change().mean()
    markdown_notes += ind_er.to_markdown()
    markdown_notes += "\n\n"
    cov_matrix = test.pct_change().apply(lambda x: np.log(1 + x)).cov()
    corr_matrix = test.pct_change().apply(lambda x: np.log(1 + x)).corr()
    markdown_notes += "corr matrix \n\n"
    markdown_notes += corr_matrix.to_markdown()
    markdown_notes += "\n\n"

    p_ret = []  # Define an empty array for portfolio returns
    p_vol = []  # Define an empty array for portfolio volatility
    p_weights = []  # Define an empty array for asset weights

    num_assets = len(test.columns)
    num_portfolios = 10000

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er)

        p_ret.append(returns)
        var = (
            cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        )  # Portfolio Variance
        sd = np.sqrt(var)  # Daily standard deviation
        ann_sd = sd * np.sqrt(250)  # Annual standard deviation = volatility
        p_vol.append(ann_sd)

    data = {"Returns": p_ret, "Volatility": p_vol}

    for counter, symbol in enumerate(test.columns.tolist()):
        # print(counter, symbol)
        data[symbol + " weight"] = [w[counter] for w in p_weights]
    portfolios = pd.DataFrame(data)

    # the minimum volatility portfolio
    min_vol_port = portfolios.iloc[portfolios["Volatility"].idxmin()]

    # Finding the optimal portfolio
    rf = 0.01  # risk factor
    optimal_risky_port = portfolios.iloc[
        ((portfolios["Returns"] - rf) / portfolios["Volatility"]).idxmax()
    ]

    markdown_notes += "optimal risk protfolio: \n\n"
    markdown_notes += optimal_risky_port.to_markdown()
    markdown_notes += "\n\n"

    # Plotting optimal portfolio
    plt.subplots(figsize=(10, 10))
    plt.scatter(
        portfolios["Volatility"], portfolios["Returns"], marker="o", s=10, alpha=0.3
    )
    plt.scatter(min_vol_port[1], min_vol_port[0], color="r", marker="*", s=500)
    plt.scatter(
        optimal_risky_port[1], optimal_risky_port[0], color="g", marker="*", s=500
    )
    fig = plt.gcf()
    if result_dir:
        fig.savefig(os.path.join(result_dir, "protfolio.png"), dpi=300)
        plt.close(fig)
        with open(os.path.join(result_dir, "protfolio.md"), "w") as f:
            f.write(markdown_notes)
    else:
        plt.show()
        return fig, markdown_notes

