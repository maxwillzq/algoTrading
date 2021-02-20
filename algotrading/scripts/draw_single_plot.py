import algotrading
from algotrading.utils import *
import pandas as pd
import pandas_datareader.data as web
import matplotlib
import mplfinance        as mpf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import datetime as dt
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)
end = dt.datetime.now()
start = dt.datetime(end.year - 1, end.month, end.day)


stock_name_list = [
    "COST",
    "TSM",
    "BABA",
    "FB",
    "AMZN",
    "AAPL",
    "GOOG",
    "NFLX",
    "AMD"
    ]
result_dir = "./save_visualization"
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
for stock_name in stock_name_list:
    df = algotrading.utils.read_stock_data_to_df(stock_name, start=start, end=end)
    print(df.columns)

    fig, axes = mpf.plot(df, 
        type='candle', 
        mav=[20, 60, 120], 
        volume=True,
        figsize=(12, 9), 
        title=stock_name,
        #savefig=stock_name + ".png",
        returnfig=True,
        #addplot=apds
        )

    # Configure chart legend and title
    axes[0].legend(["MA20", "MA60", "MA120"])
    file_name = os.path.join(result_dir, stock_name + ".png")
    fig.savefig(file_name)