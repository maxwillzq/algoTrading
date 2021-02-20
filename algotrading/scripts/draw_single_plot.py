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

logger = logging.getLogger(__name__)
end = dt.datetime.now()
start = end - dt.timedelta(days=365)
#start = dt.datetime(end.year - 1, end.month, end.day)

stock_name_list = [
    "QQQ", "SPY", "JETS", "ARKK", # ETF
    "SHOP", "BIDU", "ADBE", "U", "AMC",  # Shuping list
    "MMM",  "C", "COST", # industry stock
    "TSM", "ASML", "AMAT", # Semi
    "BABA", "FB", "AMZN", "AAPL", "GOOG", "NFLX", "AMD", "MSFT", # FANGMAN
    ]

result_dir = "./save_visualization"

def get_range_min_max(idf):
    last = len(idf)
    mav = idf['Adj Close'].rolling(20).mean()
    mav = mav[last - 30: last]
    return mav.min(), mav.max()


def main():
    parser = argparse.ArgumentParser(description="plot stock")
    parser.add_argument(
        "--name",
        required=False,
        help="The stock name list"
    )

    markdown_str = """# Daily stock plotting

    """
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    for stock_name in stock_name_list:
        df = algotrading.utils.read_stock_data_to_df(stock_name, start=start, end=end)

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
        axes[0].legend(["MA20", "MA60", "MA120"], loc="upper left")
        rmin, rmax = get_range_min_max(df)
        #axes[0].axhline(y=df['Close'].iloc[-1], color='r', linestyle='--')
        axes[0].axhline(y=rmin, color='r', linestyle='--')
        axes[0].axhline(y=rmax, color='r', linestyle='--')
        file_name = os.path.join(result_dir, stock_name + ".png")
        fig.savefig(file_name,dpi=300)
        plt.close(fig)
    
        round_df = df.round(2)
        markdown_str += "\n\\pagebreak\n\n"
        markdown_str += f"## {stock_name}\n\n"
        markdown_str += f"{round_df.tail(5).to_markdown()}\n"
        markdown_str += f"![{stock_name}]({stock_name}.png)\n\n\n"

    md_file_path = os.path.realpath(os.path.join(result_dir, "daily_plot.md"))
    with open(md_file_path, 'w') as f:
        f.write(markdown_str)

    pdf_file_path = os.path.realpath(os.path.join(result_dir, "daily_plot.pdf"))
    os.chdir(result_dir)
    output = pypandoc.convert_file(md_file_path, 'pdf', outputfile=pdf_file_path,
    extra_args=['-V', 'geometry:margin=1.5cm', '--pdf-engine=/Library/TeX/texbin/pdflatex'])

if __name__ == '__main__':
    main()
