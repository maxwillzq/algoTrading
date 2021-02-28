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
    'ADBE', 'U', 'AMC', 'BABA', 'FB', 'COST', 'CRM', 'QCOM', 'TIGR', 'ARKK',
    'AMD', 'BB', "CCL",
    "MMM",  "C", "COST",
    'TSM', 'ASML', 'AMAT',
    "BABA", "FB", "AMZN", "AAPL", "GOOG", "NFLX", "AMD", "MSFT",
    'PLTR', 'IPOE', 'SFIX'
]
etf_name_list = [
    "VTI", "DIA", "OEF", "MDY", "SPY",  "RSP", "QQQ", "QTEC", "IWB", "IWM", # Broad Market
    "MTUM", "VLUE", "QUAL", "USMV", # Factors
    "IWF", "IWD", "IVW", "IVE", # Growth of value
    "MOAT", "FFTY", "IBUY", "CIBR", "SKYY", "IPAY", "FINX", "XT", "ARKK", "BOTZ", "MOO", "ARKG", "MJ", "ARKW", "ARKQ", "PBW", "BLOK", "SNSR", # Thermatic
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
    result_str += f"- top 3 volume increase stock today: { [name for name in price_change_table_pd.nlargest(3, 'vol_change%').name] }\n"
    result_str += f"- top 3 volume decrease stock today: { [name for name in price_change_table_pd.nsmallest(3, 'vol_change%').name] }\n"
    result_str += "\n\n"
    result_str += price_change_table_pd.to_markdown()
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
        default=365,
        help="how many days, default is 300",
        type=int,
    )
    parser.add_argument(
        "--sort_by",
        default="1D%",
        help="sorted by which column.Default is 1D",
        type=str,
    )
    parser.add_argument(
        "--with_chart",
        default=False,
        help="flag control output individual stock chart",
        type=bool,
    )

    args = parser.parse_args()
    result_dir = args.result_dir
    """
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    """

    end = dt.datetime.now()
    start = end - dt.timedelta(days=args.days)

    stock_name_list = args.stock_list
    if stock_name_list == "shuping":
        stock_name_list = shuping_holding_list
    elif stock_name_list == "etf":
        stock_name_list = etf_name_list
    elif stock_name_list == "all":
        stock_name_list = default_stock_name_list
    else:
        stock_name_list = [item for item in args.stock_list.split(',')]    

    markdown_str = f"# Stock analysis report ({end})\n"
    price_change_table = []
    plotting_dict = {}
    for stock_name in stock_name_list:
        stock = algotrading.stock.Stock(stock_name)
        stock.read_stock_data(start=start, end=end)
        price_change_info = stock.get_price_change_table()
        price_change_table.append(price_change_info)

        # generate the plot if flag is true
        if args.with_chart:
            apds = []
            subplots = stock.calc_buy_sell_signal()
            apds.extend(subplots)
            stock.save_plot(result_dir, apds)
        
            round_df = stock.df.round(2)
            plotting_markdown_str = "\n\\pagebreak\n\n"
            plotting_markdown_str += f"## {stock_name}\n\n"
            plotting_markdown_str += f"{round_df.tail(5).to_markdown()}\n"
            plotting_markdown_str += f"![{stock_name}]({stock_name}.png)\n\n\n"
            plotting_dict[stock_name] = plotting_markdown_str

    # Add summary to report
    tmp_str, price_change_table_pd = generate_md_summary_from_changed_table(price_change_table, args.sort_by)
    markdown_str += tmp_str

    # add single plot to report if flag is true
    if args.with_chart:
        for ind in price_change_table_pd.index:
            key_name = price_change_table_pd.loc[ind].loc["name"]
            markdown_str += plotting_dict[key_name]
            #markdown_str += price_change_table_pd.loc[ind].to_markdown()

    # Generate markdown and pdf
    if args.stock_list == "shuping":
        output_file_name = "shuping_daily_plot"
    elif args.stock_list == "etf":
        output_file_name = "etf_daily_plot"
    else: 
        output_file_name = "daily_plot"
    md_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".md"))
    with open(md_file_path, 'w') as f:
        f.write(markdown_str)

    pdf_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".pdf"))
    os.chdir(result_dir)
    output = pypandoc.convert_file(md_file_path, 'pdf', outputfile=pdf_file_path,
    extra_args=['-V', 'geometry:margin=1.5cm', '--pdf-engine=/Library/TeX/texbin/pdflatex'])

if __name__ == '__main__':
    main()
