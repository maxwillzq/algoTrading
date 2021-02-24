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
import pypandoc
import argparse
import json
import logging
from collections import OrderedDict
import shutil

logger = logging.getLogger(__name__)
#start = dt.datetime(end.year - 1, end.month, end.day)
shuping_holding_list = [
    'ADBE', 'U', 'AMC', 'BABA', 'FB', 'COST', 'CRM', 'QCOM', 'TIGR', 'ARKK',
    'AMD',
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

def get_range_min_max(idf):
    last = len(idf)
    mav = idf['Adj Close'].rolling(20).mean().round(2)
    mav = mav[last - 30: last]
    return mav.min(), mav.max()

def calc_buy_sell_signal(df, apds):
    idf = df.copy()
    exp12     = idf['Close'].ewm(span=12, adjust=False).mean()
    exp26     = idf['Close'].ewm(span=26, adjust=False).mean()
    macd      = exp12 - exp26
    signal    = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    apds.extend([
                mpf.make_addplot(histogram,type='bar',width=0.7,panel=1,
                                color='dimgray',alpha=1,secondary_y=False),
                mpf.make_addplot(macd,panel=1,color='fuchsia',secondary_y=True),
                mpf.make_addplot(signal,panel=1,color='b',secondary_y=True),
            ])

    idf['20_EMA'] = idf['Close'].rolling(20).mean()
    idf['20_EMA_Future'] = idf['20_EMA'] + idf['20_EMA'].diff() * 5
    idf['60_EMA'] = idf['Close'].rolling(60).mean()
    idf['60_EMA_Future'] = idf['60_EMA'] + idf['60_EMA'].diff() * 5
    idf['Signal'] = 0.0  
    #idf['Signal'] = np.where(idf['20_EMA_Future'] > idf['60_EMA_Future'], 1.0, 0.0)
    idf['Signal'] = np.where(macd > signal + 0.02, 1.0, 0.0)
    idf['Position'] = idf['Signal'].diff()
    my_markers = []
    colors = []
    for i, v in idf['Position'].items():
        marker = None
        color = 'b'
        if v == 1 and idf.loc[i]["Close"] <= max(idf.loc[i]["60_EMA"], idf.loc[i]["20_EMA"]) * 1.05:
            # Buy point
            marker = '^'
            color = 'g'
            logger.debug(f"index = {i}, macd = {macd.loc[i]}, signal = {signal.loc[i]}, hist = {histogram.loc[i]}")
        elif v == -1 and idf.loc[i]["Close"] >= max(idf.loc[i]["20_EMA"],idf.loc[i]["60_EMA"]):
            # Sell point
            # marker = 'v'
            marker = None
            color = 'r'
        my_markers.append(marker)
        colors.append(color)
    apds.append(mpf.make_addplot(idf['Close'], type='scatter', marker=my_markers,markersize=45,color=colors))

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
        default="5D%",
        help="sorted by which column.Default is 5D",
        type=str,
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
        df = algotrading.utils.read_stock_data_to_df(stock_name, start=start, end=end)
        price_change_info = {}
        price_change_info["name"] = stock_name
        last = len(df) - 1
        for delta in [1, 5, 10, 20, 60, 120, 240, 500]:
            key_name = f"{delta}D%"
            if last - delta > 0:
                value = (df['Close'].iloc[last] - df['Close'].iloc[last - delta])/df['Close'].iloc[last - delta] * 100
                value = round(value, 2)
                price_change_info[key_name] = value
            else:
                price_change_info[key_name] = None

        df['20_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
        price_change_info["MA20%"] = (df['Close'].iloc[last] - df['20_EMA'].iloc[last])/df['20_EMA'].iloc[last] * 100

        price_change_table.append(price_change_info)
        apds = []

        calc_buy_sell_signal(df, apds)
        
        file_name = os.path.join(result_dir, stock_name + ".png")
        fig, axes = mpf.plot(df, 
            type='candle', 
            style="yahoo",
            mav=[20, 60, 120, 200], 
            volume=True,
            figsize=(12, 9), 
            title=stock_name,
            #savefig=stock_name + ".png",
            returnfig=True,
            volume_panel=2,
            addplot=apds,
            #hlines=[1400],
            )

        # Configure chart legend and title
        rmin, rmax = get_range_min_max(df)
        #axes[0].axhline(y=df['Close'].iloc[-1], color='r', linestyle='--')
        axes[0].axhline(y=rmin, color='r', linestyle='--')
        axes[0].axhline(y=rmax, color='r', linestyle='--')
        axes[0].legend(["MA20", "MA60", "MA120", "MA200", rmin, rmax], loc="upper left")
        fig.savefig(file_name,dpi=300)
        plt.close(fig)
    
        round_df = df.round(2)
        plotting_markdown_str = "\n\\pagebreak\n\n"
        plotting_markdown_str += f"## {stock_name}\n\n"
        plotting_markdown_str += f"{round_df.tail(5).to_markdown()}\n"
        plotting_markdown_str += f"![{stock_name}]({stock_name}.png)\n\n\n"
        plotting_dict[stock_name] = plotting_markdown_str

    # add price table
    price_change_table_md = "## price change table\n\n"
    price_change_table_pd = pd.DataFrame(price_change_table)
    price_change_table_pd = price_change_table_pd.sort_values([args.sort_by])
    #price_change_table_pd.set_index("name", inplace=True)
    price_change_table_md += price_change_table_pd.to_markdown()
    markdown_str += price_change_table_md

    # add single plot
    for ind in price_change_table_pd.index:
        #import pdb
        #pdb.set_trace()
        key_name = price_change_table_pd.loc[ind].loc["name"]
        markdown_str += plotting_dict[key_name]
        markdown_str += price_change_table_pd.loc[ind].to_markdown()

    # Generate markdown and pdf
    md_file_path = os.path.realpath(os.path.join(result_dir, "daily_plot.md"))
    with open(md_file_path, 'w') as f:
        f.write(markdown_str)

    pdf_file_path = os.path.realpath(os.path.join(result_dir, "daily_plot.pdf"))
    os.chdir(result_dir)
    output = pypandoc.convert_file(md_file_path, 'pdf', outputfile=pdf_file_path,
    extra_args=['-V', 'geometry:margin=1.5cm', '--pdf-engine=/Library/TeX/texbin/pdflatex'])

if __name__ == '__main__':
    main()
