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
from  algotrading.utils.misc_utils import generate_report_from_markdown
from  algotrading.utils.misc_utils import read_dict_from_file
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
from jinja2 import Template
from algotrading import manager

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
# start = dt.datetime(end.year - 1, end.month, end.day)

def generate_md_summary_from_changed_table(price_change_table, sort_by="1D%"):
    price_change_table_pd = pd.DataFrame(price_change_table)
    sort_by_list = sort_by.split(",")
    price_change_table_pd = price_change_table_pd.sort_values(sort_by_list)
    if len(price_change_table_pd) == 1:
        return ""
    result_str = Template("""
## price change table summary

### Quick summary:

{% for item in summary %}
- {{ item.title }}: {{ item.names }}

{{ item.table }}

{% endfor %}

### bullish stocks

{{ bullish_table }}

### undefined

{{ undefined_table }}

### bearish stocks

{{ bearish_table }}


""").render(
        summary=[
            {
                "title": "top 3 gainer today",
                "names": [name for name in price_change_table_pd.nlargest(3, "1D%").name],
                "table": price_change_table_pd.nlargest(3, "1D%").to_markdown(),
            },
            {
                "title": "top 3 loser today",
                "names": [name for name in price_change_table_pd.nsmallest(3, "1D%").name],
                "table": price_change_table_pd.nsmallest(3, "1D%").to_markdown(),
            },
            {
                "title": "top 3 volume increase stock today",
                "names": [name for name in price_change_table_pd.nlargest(3, "vol_change%").name],
                "table": price_change_table_pd.nlargest(3, "vol_change%").to_markdown(),
            },
            {
                "title": "top 3 volume decrease stock today",
                "names": [name for name in price_change_table_pd.nsmallest(3, "vol_change%").name],
                "table": price_change_table_pd.nsmallest(3, "vol_change%").to_markdown(),
            },
        ],
        bullish_table=price_change_table_pd[price_change_table_pd.mid_term == "long"].to_markdown(),
        undefined_table=price_change_table_pd[price_change_table_pd.mid_term == "undefined"].to_markdown(
        ),
        bearish_table=price_change_table_pd[price_change_table_pd.mid_term == "short"].to_markdown(
        ),
    )    
    return result_str


def get_stock_name_dict(stock_list: str) -> Dict:
    """Convert usr config stock_list into stock_name_dict

    Args:
        stock_list (str): stock_list separated by comma or file name. Example: "AMZN,GOOG".

    Returns:
        Dict[str,str]:  key value pair. key = stock symbol, value = long stock name.
    """
    result = {}
    if isinstance(stock_list, Dict):
        result = stock_list
    elif stock_list in ("shuping", "mutual_fund", "keyao", "etf", "fred"):
        result = algotrading.data.get_data_dict(f"{stock_list}.yaml")
    elif stock_list.startswith("get_"):
        method = getattr(algotrading.data, stock_list)
        result_list = method()
        for name in result_list:
            result[name] = name
    else:
        for item in stock_list.split(","):
            result[item] = item
    return result

def read_input_args(args):
    main_cf = {}
    if args.config and os.path.isfile(args.config):
        main_cf = read_dict_from_file(args.config)
    if args.extra is not None:
        extra_dict = dict(args.extra)
        for key, value in extra_dict.items():
            if value == "False":
                value = False
            elif value == "True":
                value = True
            main_cf[key] = value

    # set default value
    main_cf["days"] = int(main_cf.get("days", 250))
    main_cf["sort_by"] = main_cf.get("sort_by", "mid_term,short_term,5D%,1D%")
    main_cf["result_dir"] = main_cf.get("result_dir", os.path.abspath("./save_visualization"))
    
    return main_cf

def save_config(main_cf):
    result_dir = main_cf.get("result_dir")
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    output_config_path = os.path.join(result_dir, "input_config.yaml")
    with open(output_config_path, "w") as f:
        yaml.dump(main_cf, f)
        logger.info(f"save the config to file {output_config_path}")
        logger.info(f"next time user can rerun: at_run --config {output_config_path}")


def create_stock_object(stock_list, stock_name, stock_ticker):
    if stock_list == "fred":
        return algotrading.fred.Fred(stock_name, stock_ticker)
    return algotrading.stock.Stock(stock_name, stock_ticker)

def plot_with_mav(stock_obj, main_cf):
    days = main_cf["days"]
    if days >= 500:
        stock_obj.plot(mav=[60, 120, 240], image_name=stock_obj.name + "_long", **main_cf)
    elif days >= 250:
        stock_obj.plot(mav=[20, 60, 120], image_name=stock_obj.name + "_mid", **main_cf)
    elif days >= 60:
        stock_obj.plot(mav=[5, 10, 20], image_name=stock_obj.name + "_short", **main_cf)

def remove_png_files():
    # remove png files
    images = os.listdir(".")
    for item in images:
        if item.endswith(".png"):
            os.remove(os.path.join(".", item))

def generate_output_file_name():
    today_date = dt.datetime.now()
    date_str = today_date.strftime("%m_%d_%Y")
    return f"daily_report_{date_str}"

def generate_md_file_path(result_dir, file_name):
    return os.path.realpath(os.path.join(result_dir, file_name + ".md"))

def generate_pdf_file_path(result_dir, file_name, report_format):
    return os.path.realpath(os.path.join(result_dir, f"{file_name}.{report_format}"))


def run_main_flow(args):
    main_cf = read_input_args(args)
    save_config(main_cf)
    
    end = dt.datetime.now()
    stock_list = main_cf.get("stock_list", "shuping")
    stock_name_dict = get_stock_name_dict(stock_list)
    markdown_str = f"# Stock analysis report ({end})\n"
    
    price_change_table = []
    plotting_dict = {}
    for stock_name in stock_name_dict:
        stock = create_stock_object(stock_list, stock_name, stock_name_dict[stock_name])
        stock.read_data(**main_cf)
        stock.generate_more_data(days=14)
        price_change_info = stock.get_price_change_table()
        price_change_table.append(price_change_info)
        plot_with_mav(stock, main_cf)
        if main_cf.get("with_density", None) == "Yes":
            stock.plot_density(main_cf["result_dir"])
        plotting_dict[stock_name] = stock.to_markdown()

    markdown_str += generate_md_summary_from_changed_table(
            price_change_table, main_cf["sort_by"])

    for key_name in stock_name_dict:
        markdown_str += plotting_dict[key_name]

    # Generate markdown
    output_file_name = generate_output_file_name()
    result_dir = main_cf.get("result_dir")
    md_file_path = generate_md_file_path(result_dir, output_file_name)
    report_format = main_cf.get("report_format", "pdf")
    with open(md_file_path, "w") as f:
        f.write(markdown_str)

    try:
        pdf_file_path = os.path.realpath(
            os.path.join(result_dir, f"{output_file_name}.{report_format}")
        )
        generate_report_from_markdown(
            md_file_path, result_dir, pdf_file_path, report_format
        )
    except Exception as e:
        raise RuntimeError(
            f"Can not generate the {report_format} format report.Please read the markdown instead."
        ) from e



def main():  # type: () -> None
    """Main function to run daily plot."""
    parser = argparse.ArgumentParser(
        description="""
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
    """
    )
    default_config_file = os.path.realpath(os.path.dirname(__file__))
    default_config_file = os.path.join(default_config_file, "main_flow_template.yaml")
    parser.add_argument(
        "--config", default=default_config_file, help="set main flow config file"
    )
    parser.add_argument(
        "--extra",
        action="append",
        type=lambda kv: kv.split("="),
        dest="extra",
        help="key value pairs for all new created options. For example: --extra stock_list=PDD --extra days=250",
        required=False,
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    return run_main_flow(args)


if __name__ == "__main__":
    main()
