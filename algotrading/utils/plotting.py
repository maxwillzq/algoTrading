import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import datetime as dt
import mplfinance
import logging
import numpy as np
import os
import algotrading
from typing import Optional, Mapping

logger = logging.getLogger(__name__)


def plot_price_volume(
    df: pd.DataFrame, stock_name: Optional[str] = None, param: Optional[Mapping] = {}
):
    """
    This function creates a subplot of two graphs, the first graph shows the "Adj Close"
    value of the stock, and the second graph shows the stock's trading "Volume".
    Additionally, this function can also plot the linear regression trendline of the "Adj Close"
    values for the specified data range in the "data_range_list" parameter.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the stock data.
    stock_name (str, optional): The name of the stock to be displayed on the title of the plot.
                                If not provided, the title will be empty.
    param (dict, optional): A dictionary of optional parameters.
                            The following parameter is supported:
                            - data_range_list (list): A list of integers, representing the data range
                              in days to be used to calculate the linear regression trendline.
                              The default is [90].

    Returns:
    None

    """
    plt.figure(figsize=(12, 9))
    top = plt.subplot2grid((12, 9), (0, 0), rowspan=10, colspan=9)
    bottom = plt.subplot2grid((12, 9), (10, 0), rowspan=2, colspan=9)
    top.plot(df.index, df["Adj Close"], color="blue")  # df.index gives the dates
    top.grid()
    bottom.bar(df.index, df["Volume"])
    # set the labels
    top.axes.get_xaxis().set_visible(False)
    if stock_name:
        top.set_title(stock_name)
    top.set_ylabel("Adj Close")
    bottom.set_ylabel("Volume")
    last = len(df)
    data_range_list = [90]
    if "data_range_list" in param:
        data_range_list = param["data_range_list"]
    for data_range in data_range_list:
        rets = np.log(df["Adj Close"].iloc[last - data_range : last])
        x = df.index[last - data_range : last]
        x_ind = range(len(rets))
        slope, intercept, r_value, p_value, std_err = linregress(x_ind, rets)
        print("Linear" + str(data_range) + " = " + str(slope))
        top.plot(
            x, np.e ** (intercept + slope * x_ind), label="Linear" + str(data_range)
        )
    legend = top.legend(loc="upper left", shadow=True, fontsize="x-large")
    plt.show()


def plot_price_density(df: pd.DataFrame, param: Optional[Mapping] = {}):
    """
    Plots a density plot of the stock's adjusted close prices.

    Args:
        df (pandas.DataFrame): The dataframe containing the stock data.
        param (dict, optional): Additional parameters to customize the plot. Defaults to an empty dictionary.

    Returns:
        None

    Raises:
        None
    """
    plt.figure(figsize=(12, 9))
    ax = sns.distplot(df["Adj Close"].dropna(), bins=50, color="purple", vertical=True)
    rmin = min(df["Adj Close"]) * 0.9
    rmax = max(df["Adj Close"]) * 1.1
    step = param.get("step", 5)
    plt.yticks(np.arange(rmin, rmax, step))
    plt.grid()
    plt.show()


def plot_moving_average(
    df: pd.DataFrame,
    stock_name: str,
    param: Mapping = {"list": [20, 60, 120]},
    file_name: Optional[str] = None,
):
    """
    Plots the stock's prices along with moving averages.

    Args:
        df (pandas.DataFrame): The dataframe containing the stock data.
        stock_name (str): The name of the stock to be plotted.
        param (dict, optional): Additional parameters to customize the plot. Defaults to an empty dictionary.
        file_name (str, optional): The file name to save the plot. Defaults to None.

    Returns:
        None

    Raises:
        None
    """
    # simple moving averages
    lists = param["list"]
    if file_name:
        mplfinance.plot(
            df,
            type="candle",
            mav=lists,
            volume=True,
            figsize=(12, 9),
            title=stock_name,
            savefig=file_name,
        )
    else:
        mplfinance.plot(
            df, type="candle", mav=lists, volume=True, figsize=(12, 9), title=stock_name
        )


def plot_price_minus_moving_average(
    df: pd.DataFrame, stock_name: str, param: Mapping = {"MA": 50}
):
    """Plots the difference between stock's adjusted close price and its moving average.

    Args:
        df (pd.DataFrame): The dataframe containing the stock data.
        stock_name (str): The name of the stock.
        param (dict, optional): Additional parameters to customize the plot. Defaults to an empty dictionary.

    Returns:
        None

    Raises:
        None
    """
    ma = param["MA"]
    df[f"MA{ma}"] = df["Adj Close"].rolling(ma).mean()
    plt.plot(df.index, df["Adj Close"] - df[f"MA{ma}"])
    plt.title(f"{stock_name} Price - MA{ma}")
    plt.grid()
    plt.show()
