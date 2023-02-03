import algotrading
import algotrading.stock
from algotrading.utils import *
import pandas as pd
import pandas_datareader.data as web
import matplotlib
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import datetime as dt
import logging
import os
import numpy as np
import pypandoc
import argparse
import json
import logging
from collections import OrderedDict
from datetime import timedelta
import shutil
import algotrading
from algotrading.utils.plotting import calc_volatility, generate_portfolio

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="plot stock")
    parser.add_argument(
        "--result_dir", default="./save_visualization", help="The result dir"
    )
    parser.add_argument(
        "--stock_list",
        default="shuping",
        help="delimited stock name list",
        type=str,
    )
    parser.add_argument(
        "--days",
        default=365,
        help="how many days, default is 300",
        type=int,
    )
    parser.add_argument(
        "--sort_by",
        default="5D%",
        help="sorted by which column.Default is 5D",
        type=str,
    )
    parser.add_argument(
        "--with_chart",
        default="Yes",
        help="flag control output individual stock chart. Yes or No",
        type=str,
    )
    args = parser.parse_args()
    stock_name_list = args.stock_list
    stock_name_dict = {}
    if stock_name_list == "shuping":
        stock_name_dict = algotrading.data.get_data_dict("personal_stock_tickers.yaml")
    elif stock_name_list == "etf":
        stock_name_dict = algotrading.data.get_data_dict("etf.yaml")
    elif stock_name_list == "fred":
        stock_name_dict = algotrading.data.get_data_dict("fred.yaml")
    elif stock_name_list == "sp500":
        tmp_df = algotrading.data.get_SP500_list()
        tmp_df = tmp_df[tmp_df["GICS Sector"] == "Information Technology"]
        for index in range(len(tmp_df)):
            name = tmp_df["Symbol"].iloc[index]
            stock_name_dict[name] = name
    else:
        stock_name_list = [item for item in args.stock_list.split(",")]
        for item in stock_name_list:
            stock_name_dict[item] = item

    result_dir = args.result_dir
    # output_file_path = os.path.join(result_dir, "stock_volatility.png")
    # calc_volatility(stock_name_dict.keys(), output_file_path)
    generate_portfolio(stock_name_dict.keys(), result_dir)


if __name__ == "__main__":
    main()

# test command
# python3 portfolio_optimization.py --stock_list PDD,JD,TSM,SQ,SHOP,AMZN
