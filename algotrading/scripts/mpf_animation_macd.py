"""
This file contains a animation demo using mplfinance "external axes mode",
in which animate both the display of candlesticks as well as the display
of MACD (Moving Average Convergence Divergence) visual analysis.

In this example, instead of creating the Figure and Axes external to mplfiance,
we allow mplfinance to create the Figure and Axes using its "panel method", and
set kwarg `returnfig=True` so that mplfinance will return the Figure and Axes.

We then take those Axes and pass them back into mplfinance ("external axes mode")
as part of the animation.

Note that presently mplfinance does not support "blitting" (blitting makes animation
more efficient).  Nonetheless, the animation is efficient enough to update at least
once per second, and typically more frequently depending on the size of the plot.
"""
import pandas as pd
import mplfinance as mpf
import matplotlib.animation as animation
import algotrading
from algotrading.utils import *
import pandas as pd
import pandas_datareader.data as web
import matplotlib
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import datetime as dt
import logging
import os
import numpy as np
import pypandoc

logger = logging.getLogger(__name__)
end = dt.datetime.now()
start = end - dt.timedelta(days=365)
stock_name = "SHOP"

idf = algotrading.utils.read_stock_data_to_df(stock_name, start=start, end=end)
idf.shape
idf.head(3)
idf.tail(3)


# =======
#  MACD:

df = idf.iloc[0:30]

exp12 = df["Close"].ewm(span=12, adjust=False).mean()
exp26 = df["Close"].ewm(span=26, adjust=False).mean()
macd = exp12 - exp26
signal = macd.ewm(span=9, adjust=False).mean()
histogram = macd - signal

apds = [
    mpf.make_addplot(exp12, color="lime"),
    mpf.make_addplot(exp26, color="c"),
    mpf.make_addplot(
        histogram,
        type="bar",
        width=0.7,
        panel=1,
        color="dimgray",
        alpha=1,
        secondary_y=False,
    ),
    mpf.make_addplot(macd, panel=1, color="fuchsia", secondary_y=True),
    mpf.make_addplot(signal, panel=1, color="b", secondary_y=True),
]

s = mpf.make_mpf_style(base_mpf_style="classic", rc={"figure.facecolor": "lightgray"})

fig, axes = mpf.plot(
    df,
    type="candle",
    addplot=apds,
    figscale=1.5,
    figratio=(7, 5),
    title="\n\nMACD",
    style=s,
    volume=True,
    volume_panel=2,
    panel_ratios=(6, 3, 2),
    returnfig=True,
)

ax_main = axes[0]
ax_emav = ax_main
ax_hisg = axes[2]
ax_macd = axes[3]
ax_sign = ax_macd
ax_volu = axes[4]

df = idf


def animate(ival):
    if (20 + ival) > len(df):
        print("no more data to plot")
        ani.event_source.interval *= 3
        if ani.event_source.interval > 12000:
            exit()
        return
    data = df.iloc[0 : (30 + ival)]
    exp12 = data["Close"].ewm(span=12, adjust=False).mean()
    exp26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    apds = [
        mpf.make_addplot(exp12, color="lime", ax=ax_emav),
        mpf.make_addplot(exp26, color="c", ax=ax_emav),
        mpf.make_addplot(
            histogram, type="bar", width=0.7, color="dimgray", alpha=1, ax=ax_hisg
        ),
        mpf.make_addplot(macd, color="fuchsia", ax=ax_macd),
        mpf.make_addplot(signal, color="b", ax=ax_sign),
    ]

    for ax in axes:
        ax.clear()
    mpf.plot(data, type="candle", addplot=apds, ax=ax_main, volume=ax_volu)


ani = animation.FuncAnimation(fig, animate, interval=100)

mpf.show()
