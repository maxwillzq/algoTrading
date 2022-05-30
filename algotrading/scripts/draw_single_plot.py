import argparse
import datetime as dt
import json
import logging
import os
import shutil
import sys
from collections import OrderedDict
from datetime import timedelta
from typing import Any, Dict, List

import algotrading
import algotrading.utils
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
import yaml
from algotrading.utils import plotting
from scipy.stats import linregress

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
#start = dt.datetime(end.year - 1, end.month, end.day)

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
    tmp = price_change_table_pd[price_change_table_pd.mid_term == "long"]
    result_str += tmp.to_markdown()
    result_str += "\n\n"

    result_str += "### undefined \n\n"
    tmp = price_change_table_pd[price_change_table_pd.mid_term == "undefined"]
    result_str += tmp.to_markdown()
    result_str += "\n\n"

    result_str += "### bearish stocks \n\n"
    tmp = price_change_table_pd[price_change_table_pd.mid_term == "short"]
    result_str += tmp.to_markdown()
    result_str += "\n\n"
    return result_str, price_change_table_pd

def get_stock_name_dict(stock_list: str) -> Dict[str,str]:
    """Converrt usr config stock_list into stock_name_dict

    Args:
        stock_list (str): stock_list seperated by comma or file name. Example: "AMZN,GOOG".

    Returns:
        Dict[str,str]:  key value pair. key = stock symbol, value = long stock name.
    """
    result = {}
    if isinstance(stock_list, Dict):
        result = stock_list
    elif stock_list == "shuping":
        result = algotrading.data.get_data_dict("shuping_stock_tickers.yaml")
    elif stock_list == "401k":
        result = algotrading.data.get_data_dict("mutual_fund_tickers.yaml")
    elif stock_list == "keyao":
        result = algotrading.data.get_data_dict("keyao_stock_tickers.yaml")
    elif stock_list == "etf":
        result = algotrading.data.get_data_dict("etf_tickers.yaml")
    elif stock_list == "fred":
        result = algotrading.data.get_data_dict("fred.yaml")
    elif stock_list.startswith("get_"):
        method = getattr(algotrading.data, stock_list)
        result_list = method()
        for name in result_list:
            result[name] = name
    else:
        for item in stock_list.split(','):
            result[item] = item
    return result


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
            main_cf[key] = value

    # set default value
    if not "days" in main_cf:
        main_cf["days"] = 250
    else:
        main_cf["days"] = int(main_cf["days"])
    if not "sort_by" in main_cf:
        main_cf["sort_by"] = "mid_term,short_term,5D%,1D%"

    if not "result_dir" in main_cf:
        logger.warn("no result_dir on user input. user --extra result_dir to it")
        main_cf["result_dir"] = os.path.abspath("./save_visualization")

    result_dir = main_cf.get("result_dir")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    """
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    """
    output_config_path = os.path.join(result_dir, "input_config.yaml")
    with open(output_config_path, 'w') as f:
        yaml.dump(main_cf, f)
        logger.info(f"save the config to file {output_config_path}")
        logger.info(
            f"next time user can rerun: at_run --config {output_config_path}")

    end = dt.datetime.now()
    stock_list = main_cf.get("stock_list", "shuping")
    stock_name_dict = get_stock_name_dict(stock_list)
    markdown_str = f"# Stock analysis report ({end})\n"
    price_change_table = []
    plotting_dict = {}
    for stock_name in stock_name_dict:
        if stock_list == "fred":
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
                #main_cf['interval'] = 60
                stock.plot(
                    mav=[60, 120, 240], image_name=stock_name + "_long", **main_cf
                )
            elif main_cf["days"] >= 250:
                #main_cf['interval'] = 20
                print(main_cf)
                stock.plot(
                    mav=[20, 60, 120], image_name=stock_name + "_mid", **main_cf
                )
            elif main_cf["days"] >= 60:
                #main_cf['interval'] = 10
                stock.plot(
                    mav=[5, 10, 20], image_name=stock_name + "_short", **main_cf
                )
        #except:
        #    raise RuntimeError(f"fail to plot {stock.name}")
        if main_cf.get("with_density", None) == "Yes":
            stock.plot_density(result_dir)
        plotting_dict[stock_name] = stock.to_markdown()

    # Add summary to report
    try:
        tmp_str, price_change_table_pd = generate_md_summary_from_changed_table(
            price_change_table, main_cf["sort_by"])
        if len(price_change_table) > 1:
            markdown_str += tmp_str

        price_change_table_pd.sort_values(["buy_score"], inplace=True, ascending=False)
    except:
        pass

    # add single plot to report if flag is true
    for key_name in stock_name_dict:
        markdown_str += plotting_dict[key_name]
        #markdown_str += price_change_table_pd.loc[ind].to_markdown()

    # Generate markdown
    md_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".md"))
    with open(md_file_path, 'w') as f:
        f.write(markdown_str)

    # Generate pdf if user set
    generate_pdf = main_cf.get("generate_pdf", True)
    if generate_pdf is True:
        pdf_file_path = os.path.realpath(os.path.join(result_dir, output_file_name + ".pdf"))
        algotrading.utils.generate_pdf_from_markdown(md_file_path, result_dir, pdf_file_path)

    #remove png files
    images = os.listdir(".")
    for item in images:
        if item.endswith(".png"):
            #os.remove(os.path.join(".", item))
            pass


def main():  # type: () -> None
    '''Main function to run daily plot.'''
    parser = argparse.ArgumentParser(description='''
    plot stock. 
    the example command line:
    at_run --extra stock_list=AMZN  --extra result_dir=<result_dir>
    
    User can also define all options in config file and use config file 
    For example:
    config file content is 

    stock_list: shuping
    days: 500
    sort_by: mid_term,short_term,5D%,1D%

    run command: at_run --config config.yaml
    '''
                                     )
    default_config_file = os.path.realpath(os.path.dirname(__file__))
    default_config_file = os.path.join(
        default_config_file, "main_flow_template.yaml")
    parser.add_argument('--config', default=default_config_file,
                        help="set main flow config file")
    parser.add_argument("--extra", action='append',
                        type=lambda kv: kv.split("="), dest='extra',
                        help="key value pairs for all new created options. For example: --extra stock_list=PDD --extra days=250",
                        required=False)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    return run_main_flow(args)


if __name__ == '__main__':
    main()
