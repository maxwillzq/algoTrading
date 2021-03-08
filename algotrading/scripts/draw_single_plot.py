import algotrading
import algotrading.stock
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
import pypandoc
import argparse
import json
import logging
from collections import OrderedDict
from datetime import timedelta
import shutil

logger = logging.getLogger(__name__)
#start = dt.datetime(end.year - 1, end.month, end.day)
shuping_holding_list = [
    'ADBE', 'U', 'AMC', 'BABA', 'FB', 'COST', 'CRM', 'QCOM', 'TIGR', 'ARKK', 'ARKG',
    'AMD', 'BB', "CCL",
    "MMM",  "C", "COST", "LMT",
    'TSM', 'ASML', 'AMAT', 'PDD', 'JD',
    "BABA", "FB", "AMZN", "AAPL", "GOOG", "NFLX", "AMD", "MSFT",
    'PLTR', 'IPOE', "BEKE", "QQQ", "SPY",
]
etf_name_list = [
    "VTI", "DIA", "OEF", "MDY", "SPY",  "RSP", "QQQ", "QTEC", "IWB", "IWM", # Broad Market
    "MTUM", "VLUE", "QUAL", "USMV", # Factors
    "IWF", "IWD", "IVW", "IVE", # Gr KW", "ARKQ", "PBW", "BLOK", "SNSR", # Thermatic
    "XLC", "XLY", "XHB", "XRT", "XLP",
    "XLE", "XOP", "OIH", "TAN", "URA", 
    "XLF", "KBE", "KIE", "IAI",
    "IBB", "IHI", "IHF", "XPH",
    "XLI", "ITA", "IYT", "JETS", 
    "XLB",  
    "XME", "LIT", "REMX", "IYM",
    "XLRE", "VNQ", "VNQI", "REM", 
    "XLK", "VGT", "FDN", "SOCL", "IGV","SOXX", "XLU",
    "GLD",
    "^TNX", # US 10Y Yield
    "DX-Y.NYB", # US Dollar/USDX - Index - Cash (DX-Y.NYB)
    "USDCNY=X", # USD/CNY
    "SI=F", # Silver
    "GC=F", # Gold
    "CL=F", # Oil
    #"GDX", "XLV",
    "ARKK", "ARKG","ARKF"
    ]

default_stock_name_list = []

# User setup area: choose stock symbol list
shuping_holding_list = list(OrderedDict.fromkeys(shuping_holding_list))
etf_name_list = list(OrderedDict.fromkeys(etf_name_list))
default_stock_name_list.extend(shuping_holding_list)
default_stock_name_list.extend(etf_name_list)
default_stock_name_list = list(OrderedDict.fromkeys(default_stock_name_list))

def generate_md_summary_from_changed_table(price_change_table, sort_by="1D%"):
    price_change_table_pd = pd.DataFrame(price_change_table)
    price_change_table_pd = price_change_table_pd.sort_values([sort_by])

    result_str = ""
    result_str += "## price change table summary\n\n"
    result_str += "Quick summary:\n\n"
    result_str += f"- top 3 gainer today: { [name for name in price_change_table_pd.nlargest(3, '1D%').name] }\n"
    result_str += f"- top 3 loser today: { [name for name in price_change_table_pd.nsmallest(3, '1D%').name] }\n"
    try:
        result_str += f"- top 3 volume increase stock today: { [name for name in price_change_table_pd.nlargest(3, 'vol_change%').name] }\n"
        result_str += f"- top 3 volume decrease stock today: { [name for name in price_change_table_pd.nsmallest(3, 'vol_change%').name] }\n"
    except:
        logger.info("no volume info")
    result_str += "\n\n"
    #tmp = price_change_table_pd.drop(['name'],axis=1)
    tmp = price_change_table_pd
    result_str +=  tmp.to_markdown()
    return result_str, price_change_table_pd

def main():
    parser = argparse.ArgumentParser(description="plot stock")
    parser.add_argument(
        "--result_dir",
        default="./save_visualization",
        help="The result dir"
    )
    parser.add_argument(
        "--stock_list",
        default="shuping",
        help="delimited stock name list",
        type=str,
    )
    parser.add_argument(
        "--days",
        default=1000,
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
    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    """
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    """

    end = dt.datetime.now()
    start = end - dt.timedelta(days=args.days)
    stock_name_dict = {}

    stock_name_list = args.stock_list
    if stock_name_list == "shuping":
        stock_name_dict = algotrading.data.get_data_dict("personal_stock_tickers.yaml")
    elif stock_name_list == "etf":
        stock_name_dict = algotrading.data.get_data_dict("etf.yaml")
        stock_name_list = etf_name_list
    elif stock_name_list == "fred":
        stock_name_dict = algotrading.data.get_data_dict("fred.yaml")
    elif stock_name_list == "sp500":
        tmp_df = algotrading.data.get_SP500_list()
        for index in range(10):
            name = tmp_df['Symbol'].iloc[index]
            stock_name_dict[name] = name
    else:
        stock_name_list = [item for item in args.stock_list.split(',')]
        for item in stock_name_list:
            stock_name_dict[item] = item    

    markdown_str = f"# Stock analysis report ({end})\n"
    price_change_table = []
    plotting_dict = {}
    for stock_name in stock_name_dict:
        if stock_name_list == "fred":
            stock = algotrading.stock.Fred(stock_name, stock_name_dict[stock_name])
        else:
            stock = algotrading.stock.Stock(stock_name, stock_name_dict[stock_name])
        stock.read_data(start=start, end=end)
        price_change_info = stock.get_price_change_table()
        price_change_table.append(price_change_info)

        # generate the plot if flag is true
        if args.with_chart == "Yes":
            apds = []
            if stock_name_list != "fred":
                subplots = stock.calc_buy_sell_signal()
                apds.extend(subplots)
            stock.plot(result_dir, apds, savefig=True)        
            plotting_dict[stock_name] = stock.to_markdown()

    # Add summary to report
    tmp_str, price_change_table_pd = generate_md_summary_from_changed_table(price_change_table, args.sort_by)
    markdown_str += tmp_str

    # add single plot to report if flag is true
    if args.with_chart == "Yes":
        for ind in price_change_table_pd.index:
            key_name = price_change_table_pd.loc[ind].loc["name"]
            markdown_str += plotting_dict[key_name]
            #markdown_str += price_change_table_pd.loc[ind].to_markdown()

    # Generate markdown and pdf
    date_str = end.strftime("%m_%d_%Y")
    if args.stock_list == "shuping":
        output_file_name = f"shuping_daily_plot_{date_str}"
    elif args.stock_list == "etf":
        output_file_name = f"etf_daily_plot_{date_str}"
    elif args.stock_list == "fred":
        output_file_name = f"fred_daily_plot_{date_str}"
    elif args.stock_list == "sp500":
        output_file_name = f"sp500_daily_plot_{date_str}"
    else: 
        output_file_name = f"daily_plot_{date_str}"
    md_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".md"))
    with open(md_file_path, 'w') as f:
        f.write(markdown_str)

    pdf_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".pdf"))
    os.chdir(result_dir)
    output = pypandoc.convert_file(md_file_path, 'pdf', outputfile=pdf_file_path,
    extra_args=['-V', 'geometry:margin=1.5cm', '--pdf-engine=/Library/TeX/texbin/pdflatex'])

if __name__ == '__main__':
    main()
