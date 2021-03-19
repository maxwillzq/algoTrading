import algotrading
import algotrading.utils
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
default_stock_name_list = []

# User setup area: choose stock symbol list
def generate_md_summary_from_changed_table(price_change_table, sort_by="1D%"):
    price_change_table_pd = pd.DataFrame(price_change_table)
    sort_by_list = sort_by.split(',')
    price_change_table_pd = price_change_table_pd.sort_values(sort_by_list)

    result_str = ""
    result_str += "## price change table summary\n\n"
    result_str += "Quick summary:\n\n"
    tmp = price_change_table_pd.nlargest(3, '1D%')
    result_str += f"- top 3 gainer today: { [name for name in tmp.name] }\n\n"
    result_str += tmp.to_markdown() 
    result_str += "\n\n"

    tmp = price_change_table_pd.nsmallest(3, '1D%')
    result_str += f"- top 3 loser today: { [name for name in tmp.name] }\n\n"
    result_str += tmp.to_markdown() 
    result_str += "\n\n"

    try:
        tmp = price_change_table_pd.nlargest(3, 'vol_change%')
        result_str += f"- top 3 volume increase stock today: { [name for name in tmp.name] }\n\n"
        result_str += tmp.to_markdown() 
        result_str += "\n\n"
        tmp = price_change_table_pd.nsmallest(3, 'vol_change%')
        result_str += f"- top 3 volume decrease stock today: { [name for name in tmp.name] }\n\n"
        result_str += tmp.to_markdown() 
        result_str += "\n\n"
    except:
        logger.info("no volume info")
    result_str += "\n\nFull table\n\n"
    #tmp = price_change_table_pd.drop(['name'],axis=1)
    result_str += "### bullish stocks \n\n"
    tmp = price_change_table_pd[price_change_table_pd.mid_term == "bullish"]
    result_str +=  tmp.to_markdown()
    result_str += "\n\n"
    
    result_str += "### undefined \n\n"
    tmp = price_change_table_pd[price_change_table_pd.mid_term == "undefined"]
    result_str +=  tmp.to_markdown()
    result_str += "\n\n"

    result_str += "### bearish stocks \n\n"
    tmp = price_change_table_pd[price_change_table_pd.mid_term == "bearish"]
    result_str +=  tmp.to_markdown()
    result_str += "\n\n"
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
        default=250,
        help="how many days, default is 300",
        type=int,
    )
    parser.add_argument(
        "--sort_by",
        default="mid_term,short_term,5D%,1D%",
        help="sorted by which column.Default is 5D",
        type=str,
    )
    parser.add_argument(
        "--with_chart",
        default="Yes",
        help="flag control output individual stock chart. Yes or No",
        type=str,
    )

    parser.add_argument(
        "--with_density",
        default="No",
        help="flag control output individual stock density chart. Yes or No",
        type=str,
    )

    parser.add_argument(
        "--pivot_limit",
        default=1.5,
        help="flag control output pivot lines",
        type=float,
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
    stock_name_dict = {}

    stock_name_list = args.stock_list
    date_str = end.strftime("%m_%d_%Y")
    output_file_name = f"{stock_name_list}_daily_plot_{date_str}"
    if stock_name_list == "shuping":
        stock_name_dict = algotrading.data.get_data_dict("personal_stock_tickers.yaml")
    elif stock_name_list == "keyao":
        stock_name_dict = algotrading.data.get_data_dict("keyao_stock_tickers.yaml")
    elif stock_name_list == "etf":
        stock_name_dict = algotrading.data.get_data_dict("etf.yaml")
    elif stock_name_list == "fred":
        stock_name_dict = algotrading.data.get_data_dict("fred.yaml")
    elif stock_name_list.startswith("get"):
        method = getattr(algotrading.data, stock_name_list)
        result_list = method()
        for name in result_list:
            stock_name_dict[name] = name
    else:
        stock_name_list = [item for item in args.stock_list.split(',')]
        for item in stock_name_list:
            stock_name_dict[item] = item
        output_file_name = f"daily_plot_{date_str}"    

    markdown_str = f"# Stock analysis report ({end})\n"
    price_change_table = []
    plotting_dict = {}
    for stock_name in stock_name_dict:
        if stock_name_list == "fred":
            stock = algotrading.fred.Fred(stock_name, stock_name_dict[stock_name])
        else:
            stock = algotrading.stock.Stock(stock_name, stock_name_dict[stock_name])
        stock.read_data(days=args.days)
        price_change_info = stock.get_price_change_table()
        price_change_table.append(price_change_info)

        # generate the plot if flag is true
        if args.with_chart == "Yes":
            apds = []
            if stock_name_list != "fred":
                subplots = stock.calc_buy_sell_signal()
                apds.extend(subplots)
            try:
                if args.days >= 250:
                    stock.plot(result_dir, apds, 
                    mav=[20, 60, 120], image_name=stock_name + "_long"
                    )
                else:
                    stock.plot(result_dir, apds, 
                    mav=[5, 10, 20], image_name=stock_name + "_short",
                    add_pivot=True,
                    pivot_limit=args.pivot_limit
                    )
            except:
                raise RuntimeError(f"fail to plot {stock.name}") 
            if args.with_density == "Yes":
                stock.plot_density(result_dir)     
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
    md_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".md"))
    with open(md_file_path, 'w') as f:
        f.write(markdown_str)

    pdf_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".pdf"))
    os.chdir(result_dir)
    output = pypandoc.convert_file(md_file_path, 'pdf', outputfile=pdf_file_path,
    extra_args=['-V', 'geometry:margin=1.5cm', '--pdf-engine=/Library/TeX/texbin/pdflatex'])

    #remove png files    
    images = os.listdir(".")
    for item in images:
        if item.endswith(".png"):
            #os.remove(os.path.join(".", item))
            pass


if __name__ == '__main__':
    main()
