from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import datetime as dt
import mplfinance as mpf
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

class Stock:
    def __init__(self, stock_name):
        self.stock_name = stock_name
        self.markdown_notes = []

    def read_stock_data(self, start = None, end = None):
        """Read Stock Data from Yahoo Finance
        """
        if end is None:
            end = dt.datetime.now()
        logger.info(f"today is {end}")
        if not start:
            start = dt.datetime(end.year-2, end.month, end.day)
        df = web.DataReader(self.stock_name, 'yahoo', start, end)
        df.index = pd.to_datetime(df.index)
        self.df = df
        return self.df
    
    def get_ma_range_min_max(self, MA=20):
        last = len(self.df)
        mav = self.df['Adj Close'].rolling(MA).mean().round(2)
        mav = mav[last - MA: last]
        return mav.min(), mav.max()
    
    def plot_band_lines(self, step=300):
        apds = []
        idf = self.df.copy()
        idf['Date'] = pd.to_datetime(idf.index)
        idf['Date'] = idf['Date'].map(dt.datetime.toordinal)
        
        data1 = idf.copy()
        data1 = data1[len(data1)-step: len(data1)-1]

        # high trend line
        while len(data1)>10:
            reg = linregress(
                            x= data1['Date'],
                            y=data1['High'],
            )
            data1 = data1.loc[data1['High'] > reg[0] * data1['Date'] + reg[1]]
        reg = linregress(
                        x= data1['Date'],
                        y=data1['High'],
        )

        idf['high_trend'] = reg[0] * idf['Date'] + reg[1]

        # low trend line
        data1 = idf.copy()
        data1 = data1[len(data1)-step: len(data1)-1]

        while len(data1)>10:
            reg = linregress(
                            x= data1['Date'],
                            y=data1['Low'],
                            )
            data1 = data1.loc[data1['Low'] < reg[0] * data1['Date'] + reg[1]]

        reg = linregress(
                            x= data1['Date'],
                            y=data1['Low'],
                            )
        idf['low_trend'] = reg[0] * idf['Date'] + reg[1]

        #idf['Close'].plot()
        #idf['high_trend'].plot()
        #idf['low_trend'].plot()
        #idf["Prediction"].plot()
        #plt.show()

        apds.extend([
                    mpf.make_addplot(idf["high_trend"],type="line", marker='^', color='r'),
                    mpf.make_addplot(idf["low_trend"], color='r')])
        return apds

    def plot_trend_line(self):
        idf = self.df.copy()
        idf['Date'] = pd.to_datetime(idf.index)
        idf['Date'] = idf['Date'].map(dt.datetime.toordinal)
        data1 = idf.copy()

        rets = np.log(data1['Close'])
        x = data1["Date"]
        slope, intercept, r_value, p_value, std_err = linregress(x, rets)
        idf['Prediction'] = np.e ** (intercept + slope * idf["Date"])

        result = {}
        for days in [365, 2 * 365, 3 * 365]:
            predict_price = np.e ** (intercept + slope * (idf['Date'].iloc[-1] + days) )
            predict_price = round(predict_price, 2)
            result[days] = predict_price
        
        #idf['Close'].plot()
        #idf["Prediction"].plot()
        #plt.show()
        apds = []
        apds.extend([
                    mpf.make_addplot(idf["Prediction"], type="scatter")
                    ])
        return apds, result

    def calc_buy_sell_signal(self):
        idf = self.df.copy()
        exp12     = idf['Close'].ewm(span=12, adjust=False).mean()
        exp26     = idf['Close'].ewm(span=26, adjust=False).mean()
        macd      = exp12 - exp26
        signal    = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        apds = []
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
        index = 0
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
            # 抵扣价使用黄色
            if len(idf) - 1 - 20 == index or len(idf) - 1 - 60 == index:
                marker = '*'
                color = 'y'
            
            my_markers.append(marker)
            colors.append(color)
            index += 1
        apds.append(mpf.make_addplot(idf['Close'], type='scatter', marker=my_markers,markersize=45,color=colors))
        return apds
    
    def get_price_change_table(self):
        """generate result_dict dict.
        key is "5D%, 10D% ..." or moving average "20MA% .."
        value is difference to this baseline
        """
        result_dict = {}
        result_dict["name"] = self.stock_name
        last = len(self.df) - 1
        for delta in [1, 5, 20, 60, 120, 240]:
            key_name = f"{delta}D%"
            if last - delta > 0:
                value = (self.df['Close'].iloc[last] - self.df['Close'].iloc[last - delta])/self.df['Close'].iloc[last - delta] * 100
                value = round(value, 2)
                result_dict[key_name] = value
            else:
                result_dict[key_name] = None

        for delta in [20, 60, 120]:
            key_name = f"{delta}MA%"
            df_EMA = self.df['Close'].rolling(delta).mean().round(2)
            value = (self.df['Close'].iloc[last] - df_EMA.iloc[last])/df_EMA.iloc[last] * 100
            value = round(value, 2)
            result_dict[key_name] = value

        # about volume
        df_volume_EMA = self.df['Volume'].rolling(20).mean().round(2)
        key_name = "vol_change%"
        value = (self.df['Volume'].iloc[last] - df_volume_EMA.iloc[last])/df_volume_EMA.iloc[last] * 100
        value = round(value, 2)
        result_dict[key_name] = value
        
        return result_dict
    
    def save_plot(self, result_dir, apds=[]):
        file_name = os.path.join(result_dir, self.stock_name + ".png")
        mav = [20, 60, 120, 200]
        legend_names = [f"MA{item}" for item in mav]
        last = len(self.df) - 1
        delta = 1
        daily_percentage = (self.df['Close'].iloc[last] - self.df['Close'].iloc[last - delta])/self.df['Close'].iloc[last - delta] * 100
        daily_percentage = round(daily_percentage, 2)
        fig, axes = mpf.plot(self.df, 
            type='candle', 
            style="yahoo",
            mav=mav, 
            volume=True,
            figsize=(12, 9), 
            title=f"Today's increase={daily_percentage}%",
            returnfig=True,
            volume_panel=2,
            addplot=apds,
            )

        # Configure chart legend and title
        rmin, rmax = self.get_ma_range_min_max(MA=20)
        rmin = round(rmin, 2)
        rmax = round(rmax, 2)
        axes[0].axhline(y=rmin, color='r', linestyle='--')
        axes[0].axhline(y=rmax, color='r', linestyle='--')
        legend_names.extend([rmin, rmax])
        axes[0].legend(legend_names, loc="upper left")
        fig.savefig(file_name,dpi=300)
        plt.close(fig)
        return file_name
    
    




