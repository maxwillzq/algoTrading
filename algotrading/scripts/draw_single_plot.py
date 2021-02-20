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

logger = logging.getLogger(__name__)
end = dt.datetime.now()
start = end - dt.timedelta(days=365)
#start = dt.datetime(end.year - 1, end.month, end.day)
shuping_holding_list = ['ADBE', 'U', 'AMC', 'BABA', 'FB', 'COST'] 
etf_name_list = [
    "VTI", "DIA", "OEF", "MDY", "SPY",  "RSP", "QQQ", "QTEC", "IWB", "IWM", # Broad Market
    "MTUM", "VLUE", "QUAL", "USMV", # Factors
    "IWF", "IWD", "IVW", "IVE", # Growth of value
    "MOAT", "FFTY", "IBUY", "CIBR", "SKYY", "IPAY", "FINX", "XT", "ARKK", "BOTZ", "MOO", "ARKG", "MJ", "ARKW", "ARKQ", "PBW", "BLOK", "SNSR", # Thermatic
    "XLC", "XLY", "XHB", "XRT", "XLP",
    "XLE", "XOP", "OIH", "TAN", "URA", 
    "XLF", "KBE", "KIE", "IAI",
    "XLV", "IBB", "IHI", "IHF", "XPH",
    "XLI", "ITA", "IYT", "JETS", 
    "XLB", "GDX", "XME", "LIT", "REMX", "IYM",
    "XLRE", "VNQ", "VNQI", "REM", 
    "XLK", "VGT", "FDN", "SOCL", "IGV","SOXX", "XLU"]
industry_stock_list = ["MMM",  "C", "COST"]
semi_stock_list = ['TSM', 'ASML', 'AMAT']
fangman_stock_list = ["BABA", "FB", "AMZN", "AAPL", "GOOG", "NFLX", "AMD", "MSFT"]

stock_name_list = []

# User setup area: choose stock symbol list
stock_name_list.extend(shuping_holding_list)
stock_name_list.extend(etf_name_list)
stock_name_list.extend(industry_stock_list)
stock_name_list.extend(semi_stock_list)
stock_name_list.extend(fangman_stock_list)

stock_name_list = list(OrderedDict.fromkeys(stock_name_list)) 

result_dir = "./save_visualization"
if os.path.isdir(result_dir):
    os.rmdir(result_dir)
os.mkdir(result_dir)

def get_range_min_max(idf):
    last = len(idf)
    mav = idf['Adj Close'].rolling(20).mean().round(2)
    mav = mav[last - 30: last]
    return mav.min(), mav.max()


def main():
    parser = argparse.ArgumentParser(description="plot stock")
    parser.add_argument(
        "--name",
        required=False,
        help="The stock name list"
    )

    markdown_str = "# Daily stock plotting\n"
    price_change_table = []
    plotting_dict = {}
    for stock_name in stock_name_list:
        df = algotrading.utils.read_stock_data_to_df(stock_name, start=start, end=end)
        price_change_info = {}
        price_change_info["name"] = stock_name
        last = len(df) - 1
        for delta in [1, 5, 10, 20, 60, 120]:
            key_name = f"{delta}D%"
            value = (df['Close'].iloc[last] - df['Close'].iloc[last - delta])/df['Close'].iloc[last - delta] * 100
            value = round(value, 2)
            price_change_info[key_name] = value
        price_change_table.append(price_change_info)

        mc = mpf.make_marketcolors(up='g',down='r')
        john  = mpf.make_mpf_style(base_mpf_style='charles', marketcolors=mc, mavcolors=['b', 'g', 'r'])

        exp12     = df['Close'].ewm(span=12, adjust=False).mean()
        exp26     = df['Close'].ewm(span=26, adjust=False).mean()
        macd      = exp12 - exp26
        signal    = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        apds = [
                #mpf.make_addplot(exp12,color='lime'),
                #mpf.make_addplot(exp26,color='c'),
                mpf.make_addplot(histogram,type='bar',width=0.7,panel=1,
                                color='dimgray',alpha=1,secondary_y=False),
                mpf.make_addplot(macd,panel=1,color='fuchsia',secondary_y=True),
                mpf.make_addplot(signal,panel=1,color='b',secondary_y=True),
            ]
        
        file_name = os.path.join(result_dir, stock_name + ".png")
        if not os.path.isfile(file_name):
            fig, axes = mpf.plot(df, 
                type='candle', 
                style="yahoo",
                mav=[20, 60, 120], 
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
            axes[0].legend(["MA20", "MA60", "MA120", rmin, rmax], loc="upper left")
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
    price_change_table_pd = price_change_table_pd.sort_values(["120D%", "60D%", "20D%"], ascending=[False, False, False])
    price_change_table_pd.set_index("name", inplace=True)
    price_change_table_md += price_change_table_pd.to_markdown()
    markdown_str += price_change_table_md

    # add single plot
    for ind in price_change_table_pd.index:
        key_name = ind
        markdown_str += plotting_dict[key_name]
        markdown_str += price_change_table_pd[ind].to_markdown()

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
