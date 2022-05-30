from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime as dt
import logging
import os
from typing import KeysView

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_ta as ta
import seaborn as sns
import yahoo_fin.stock_info as si
import yfinance as yf
from matplotlib.axes import Axes
from scipy.stats import linregress

import algotrading
from algotrading import stock_base

logger = logging.getLogger(__name__)


class Stock(stock_base.StockBase):
    def __init__(self, name, description=None, **kwargs):
        super().__init__(name, description, **kwargs)

    def plot(self, **kwargs):
        plot_style = 'plot_style_1'
        if 'plot_style' in kwargs:
            plot_style = kwargs['plot_style']
        method = getattr(self, plot_style)
        return method(**kwargs)

    def plot_style_1(self, **kwargs):
        try:
            self.add_quick_summary()
        except:
            pass
        #self.add_notes()

        try:
            self.plot_earning(**kwargs)
        except:
            pass

        df = self.df.copy()
        # add moving average
        mav = [20, 60, 120]
        if 'mav' in kwargs:
            mav = kwargs['mav']
        colors = ['black', 'red', 'blue', 'yellow']
        legend_names = []
        added_plots = []
        for index in range(len(mav)):
            item = mav[index]
            color = colors[index]
            if len(df) <= item:
                break
            df_sma = df[f"SMA{item}"]
            df_ema = df[f"EMA{item}"]
            added_plots.append(
                mpf.make_addplot(df_sma, color=color)
            )
            legend_names.append(f"SMA{item}")
            added_plots.append(
                mpf.make_addplot(df_ema, color=color, linestyle='--')
            )
            legend_names.append(f"EMA{item}")

        # add MACD
        added_plots.extend([
            mpf.make_addplot(self.df["MACD_OSC"], type='bar', width=0.7, panel=1,
                             color='dimgray', alpha=1, secondary_y=False),
            mpf.make_addplot(self.df["MACD_DIF"], panel=1, color='fuchsia', secondary_y=False),
            mpf.make_addplot(self.df["MACD_DEM"], panel=1, color='b', secondary_y=True),
        ])

        # add 抵扣价
        my_markers = []
        colors = []
        for index in range(len(self.df)):
            marker = None
            color = 'b'
            # 抵扣价使用黄色
            if len(self.df) - 1 - 20 == index or len(self.df) - 1 - 60 == index or len(self.df) - 1 - 120 == index:
                marker = '*'
                color = 'y'
            my_markers.append(marker)
            colors.append(color)
        added_plots.append(mpf.make_addplot(
            self.df["Close"], type='scatter', marker=my_markers, markersize=45, color=colors))

        # add bias ratio
        bias_ratio = self.df['bias_ratio']
        added_plots.extend(
            [
                mpf.make_addplot(bias_ratio, panel=3, type='bar', width=0.7,
                                 color='b', ylabel="bias_ratio", secondary_y=False),
            ]
        )

        # add rsi
        added_plots.extend([
            mpf.make_addplot(self.df["RSI"], panel=4, color='fuchsia',
                             ylabel="RSI", secondary_y=False),
            mpf.make_addplot(self.df["MFI"], panel=4, color='black',
                             ylabel="MFI(black),RSI", secondary_y=False),
            mpf.make_addplot([70] * len(self.df), panel=4, color='r',
                             linestyle='--', secondary_y=False),
            mpf.make_addplot([30] * len(self.df), panel=4, color='r',
                             linestyle='--', secondary_y=False),
            mpf.make_addplot([50] * len(self.df), panel=4, color='g',
                             linestyle='--', secondary_y=False),
        ]),

        # add title
        last = len(df) - 1
        daily_percentage = (df["Close"].iloc[last] - df["Close"].iloc[last - 1]
                            ) / df["Close"].iloc[last - 1] * 100
        daily_percentage = round(daily_percentage, 2)

        d1 = df.index[0]
        d2 = df.index[-1]
        tdates = [(d1, d2)]

        fig, axes = mpf.plot(df,
                             type='candle',
                             style="yahoo",
                             volume=True,
                             figsize=(12, 9),
                             title=f"{self.name} today increase={daily_percentage}%",
                             returnfig=True,
                             volume_panel=2,
                             addplot=added_plots,
                             #tlines=[
                             #    dict(tlines=tdates,tline_use=['open','close',#'high','low'],tline_method='least-squares',#colors='black')],
                             )

        # Get pivot and plot

        #if 'pivot_type' in kwargs and kwargs['pivot_type'] is not None:
        result = self.get_pivot(**kwargs)
        result.sort()
        colors = ['r', 'g', 'b', 'y']
        i = 0
        for pivot in result:
            if len(result) % 2 == 1 and i == len(result) // 2:
                linestyle = "-"
                color = 'y'
            elif i <= len(result) // 2:
                linestyle = "--"
                color = 'r'
            elif i > len(result) // 2:
                linestyle = "--"
                color = 'g'
            axes[0].axhline(y=pivot, color=color, linestyle=linestyle)
            i = i + 1
            legend_names.append(pivot)

        axes[0].legend(legend_names, loc="upper left")

        # save to file if possible
        image_name = self.name
        if "image_name" in kwargs:
            image_name = kwargs['image_name']
        result_dir = kwargs['result_dir'] if 'result_dir' in kwargs else None
        if result_dir is not None:
            file_name = os.path.join(result_dir, image_name + ".png")
            fig.savefig(file_name, dpi=300)
            plt.close('all')
            self.markdown_notes += "\n\n \pagebreak\n\n"
            self.markdown_notes += f"![{image_name}]({image_name}.png)\n\n\n"
            return file_name
        else:
            return fig, axes
