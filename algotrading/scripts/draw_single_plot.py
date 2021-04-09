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
import sys

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
    try:
        tmp = price_change_table_pd.nlargest(3, '1D%')
        result_str += f"- top 3 gainer today: { [name for name in tmp.name] }\n\n"
        result_str += tmp.to_markdown() 
        result_str += "\n\n"

        tmp = price_change_table_pd.nsmallest(3, '1D%')
        result_str += f"- top 3 loser today: { [name for name in tmp.name] }\n\n"
        result_str += tmp.to_markdown() 
        result_str += "\n\n"
    except:
        pass

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

def run_main_flow(args):
    main_cf = {}
    if args.config and os.path.isfile(args.config):
        main_cf = algotrading.utils.read_dict_from_file(args.config)
    if args.extra is not None:
        extra_dict = dict(args.extra)
        for key in extra_dict:
            value = extra_dict[key]
            if value == "False":
                value = False
            elif value == "True":
                value = True
            main_cf[key] =  value

    # set default value
    if not "days" in main_cf:
        main_cf["days"] = 250
    else:
        main_cf["days"] = int(main_cf["days"])
    if not "sort_by" in main_cf:
        main_cf["sort_by"] = "mid_term,short_term,5D%,1D%"
 
    result_dir = main_cf.get("result_dir","./save_visualization")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    """
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    """

    end = dt.datetime.now()
    stock_name_dict = {}

    stock_name_list = main_cf.get("stock_list", "shuping")
    date_str = end.strftime("%m_%d_%Y")
    output_file_name = f"{stock_name_list}_daily_plot_{date_str}"
    if stock_name_list == "shuping":
        stock_name_dict = algotrading.data.get_data_dict("personal_stock_tickers.yaml")
    elif stock_name_list == "401k":
        stock_name_dict = algotrading.data.get_data_dict("mutual_fund_tickers.yaml")
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
        stock_name_list = [item for item in stock_name_list.split(',')]
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
        stock.read_data(**main_cf)
        stock.generate_more_data(days=14)

        price_change_info = stock.get_price_change_table()
        price_change_table.append(price_change_info)
        # generate the plot if flag is true
        #try:
        if True:
            if main_cf["days"] >= 500:
                stock.plot(result_dir=result_dir,
                mav=[60, 120, 240], image_name=stock_name + "_long", **main_cf
                )
            elif main_cf["days"] >= 250:
                stock.plot(result_dir=result_dir,
                mav=[20, 60, 120], image_name=stock_name + "_mid", **main_cf
                )
            elif main_cf["days"] >= 60:
                stock.plot(result_dir=result_dir,
                mav=[5, 10, 20], image_name=stock_name + "_short", **main_cf
                )
        #except:
        #    raise RuntimeError(f"fail to plot {stock.name}") 
        if main_cf.get("with_density", None) == "Yes":
            stock.plot_density(result_dir)     
        plotting_dict[stock_name] = stock.to_markdown()

    
    # Add summary to report
    tmp_str, price_change_table_pd = generate_md_summary_from_changed_table(price_change_table, main_cf["sort_by"])
    markdown_str += tmp_str

    # add single plot to report if flag is true
    price_change_table_pd.sort_values(["buy_score"], inplace=True, ascending=False)
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

def main():  # type: () -> None
    parser = argparse.ArgumentParser(description="plot stock")
    default_config_file = os.path.realpath(os.path.dirname(__file__))
    default_config_file = os.path.join(
        default_config_file, "main_flow_template.yaml")
    parser.add_argument('--config', default=default_config_file,
                        help="set main flow config file")
    parser.add_argument("--extra", action='append',
                        type=lambda kv: kv.split("="), dest='extra',
                        help="key value pairs for all new created options. For example: --extra python_usr_dir=<>",
                        required=False)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    return run_main_flow(args)


if __name__ == '__main__':
    main()
