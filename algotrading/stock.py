from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from typing import KeysView
from matplotlib.axes import Axes
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
import yfinance as yf
import yahoo_fin.stock_info as si
import pandas_ta as ta
import algotrading
logger = logging.getLogger(__name__)

def _convert_to_numeric(s):
    """
    Convert str to number
    """
    def force_float(elt):
        elt = elt.replace(',','')

        try:
            return float(elt)
        except:
            return elt
    if isinstance(s, float) or isinstance(s, int):
        return s
    if s is None:
        return s
    if "M" in s:
        s = s.strip("M")
        return force_float(s) * 1_000_000
    if "B" in s:
        s = s.strip("B")
        return force_float(s) * 1_000_000_000
    if '%' in s:
        s = s.strip("%")
        return force_float(s) /100.0
    
    return force_float(s)

class Stock:
    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        self.markdown_notes = ""
        #self.markdown_notes = "\n\\pagebreak\n\n"
        title = self.description if self.description else self.name
        self.markdown_notes += f"## {title}\n\n"
        self.df = None
        self.attribute = {}
    
    def is_good_business(self):
        """
        define the checker function for good busniess:
        - Good profit margin. Margin > 10%
        - Revenue keep grow. YOY rate > 20%
        - Revenue > 0.1 Billion USD
        - Positive cash flow.
        """
        status = True
        result = {}
        message = []
        stats = si.get_stats(self.name)
        logger.debug(stats)
        for index in range(len(stats)):
            result[stats.Attribute.iloc[index]] = _convert_to_numeric(stats.Value.iloc[index])
        
        # check rules
        if result["Profit Margin"] < 0.1:
            message.append(f"fail: weak profit margin. profit margin = {result['Profit Margin']*100}")
            status = False
        if result["Revenue (ttm)"] < 100000000:
            message.append(f"fail: revenue less than 0.1 * billion." + 'Revenue = ' + str(result["Revenue (ttm)"]/1000000000) + "Billion")
            status = False
        if result["Levered Free Cash Flow (ttm)"] < 0:
            message.append(f"fail: free cash flow is negative, Levered free cash flow (ttm) = " + str(result["Levered Free Cash Flow (ttm)"]/1000000000) + "Billion USD")
            status = False
        if result["Quarterly Revenue Growth (yoy)"] < 0.1:
            message.append(f"fail: revenue growth yoy less than 10%." + "rate = " + str(result["Quarterly Revenue Growth (yoy)"] * 100))
            status = False
        logger.info(message)
        result_dict = result
        return status, message, result_dict

    def read_data(self, *args, **kwargs):
        """Read Stock Data from Yahoo Finance
        """
        end = dt.datetime.now()
        start =  end - dt.timedelta(kwargs.get("days", 365))
        try:
            df = web.DataReader(self.name, 'yahoo', start, end)
        except:
            logger.error(f"fail to get data for symbol {self.name}")
            raise RuntimeError()
        df.index = pd.to_datetime(df.index)
        df.drop(columns=['Adj Close'], inplace=True)

        #add shift if need
        if 'shift' in kwargs:
            shift = int(kwargs['shift'])
            for item in ["High", "Low", "Open", "Close", "Volume"]:
                last_day_value = df[item].iloc[-1]
                df[item] = df[item].shift(shift, fill_value=last_day_value)
            self.attribute["virtual_shift"] = shift

        volume_mean = df['Volume'].mean()
        df['normalized_volume'] = df['Volume']/volume_mean
        # Generate moving average data
        df["SMA5"] = df["Close"].rolling(5).mean().round(2)
        df["SMA10"] = df["Close"].rolling(10).mean().round(2)
        df["SMA20"] = df["Close"].rolling(20).mean().round(2)
        df["SMA60"] = df["Close"].rolling(60).mean().round(2)
        df["SMA120"] = df["Close"].rolling(120).mean().round(2)
        df["SMA240"] = df["Close"].rolling(240).mean().round(2)
        df['EMA5'] = df["Close"].ewm(span=5, adjust=False).mean().round(2)
        df['EMA10'] = df["Close"].ewm(span=10, adjust=False).mean().round(2)
        df['EMA20'] = df["Close"].ewm(span=20, adjust=False).mean().round(2)
        df['EMA60'] = df["Close"].ewm(span=60, adjust=False).mean().round(2)
        df['EMA120'] = df["Close"].ewm(span=120, adjust=False).mean().round(2)
        df['EMA240'] = df["Close"].ewm(span=240, adjust=False).mean().round(2)

        # MACD, see https://zh.wikipedia.org/wiki/%E6%8C%87%E6%95%B0%E5%B9%B3%E6%BB%91%E7%A7%BB%E5%8A%A8%E5%B9%B3%E5%9D%87%E7%BA%BF
        df['MACD_DIF'] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean() # macd
        df['MACD_DEM'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean() # signal
        df['MACD_OSC'] = df['MACD_DIF'] - df['MACD_DEM'] # histgram

        # get stock information
        stock_info = yf.Ticker(self.name)
        try:
            self.attribute.update(stock_info.info)
        except:
            logger.error("fail to run stock_info.info func")

        try:
            date = si.get_next_earnings_date(self.name)
            self.attribute["next_earnings_date"] = date
        except:
            logger.warn("can not get next_earning date")
        tmp = df[
            ["Close", "normalized_volume", 
            "SMA5", "SMA10", "SMA20",
            "SMA60", "SMA120","SMA240"]
            ].tail(5).to_markdown()
        self.markdown_notes += f"\n\n{tmp}\n\n"

        self.df = df
        return self.df
    
    def generate_more_data(self, days=14):
        # Generate more data
        self.calc_average_true_range(days=days) #  ATR
        self.calc_relative_strength_index(days=days)  # RSI
        self.calc_momentum_indicator(days=days) # Momentum
        self.calc_william_ratio(days=days) # W%R
        self.calc_money_flow_index(days=days) #MFI
        self.calc_bias_ratio(days=60) #bias_ratio
        self.calc_bull_bear_signal()
        self.calc_buy_sell_signal()

        # Generate delta ratio
        last = len(self.df) - 1
        for delta in [1, 5, 20]:
            key_name = f"{delta}D%"
            if last - delta > 0:
                value = (self.df["Close"].iloc[last] - self.df["Close"].iloc[last - delta])/self.df["Close"].iloc[last - delta] * 100
                value = round(value, 2)
                self.attribute[key_name] = value
            else:
                self.attribute[key_name] = None

    def calc_bull_bear_signal(self):
        df = self.df
        # short-term signal
        if df["SMA5"].iloc[-1] > df["SMA10"].iloc[-1] > df["SMA20"].iloc[-1]:
            self.attribute["short_term"] = "long"
        elif df["SMA5"].iloc[-1] < df["SMA10"].iloc[-1] < df["SMA20"].iloc[-1]:
            self.attribute["short_term"] = "short"
        else:
            self.attribute["short_term"] = "undefined"
        
        if df["SMA5"].iloc[-2] > df["SMA10"].iloc[-2] > df["SMA20"].iloc[-2]:
            self.attribute["yesterday_short_term"] = "long"
        elif df["SMA5"].iloc[-2] < df["SMA10"].iloc[-2] < df["SMA20"].iloc[-2]:
            self.attribute["yesterday_short_term"] = "short"
        else:
            self.attribute["yesterday_short_term"] = "undefined"

        # mid-term signal
        if df["SMA20"].iloc[-1] > df["SMA60"].iloc[-1] > df["SMA120"].iloc[-1]:
            self.attribute["mid_term"] = "long"
        elif df["SMA20"].iloc[-1] < df["SMA60"].iloc[-1] < df["SMA120"].iloc[-1]:
            self.attribute["mid_term"] = "short"
        else:
            self.attribute["mid_term"] = "undefined"

        # long-term signal
        if df["SMA60"].iloc[-1] > df["SMA120"].iloc[-1] > df["SMA240"].iloc[-1]:
            self.attribute["long_term"] = "long"
        elif df["SMA60"].iloc[-1] < df["SMA120"].iloc[-1] < df["SMA240"].iloc[-1]:
            self.attribute["long_term"] = "short"
        else:
            self.attribute["long_term"] = "undefined"
        return self.attribute
        
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
        self.add_notes()
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
                mpf.make_addplot(self.df["MACD_OSC"],type='bar',width=0.7,panel=1,
                                    color='dimgray',alpha=1,secondary_y=False),
                mpf.make_addplot(self.df["MACD_DIF"], panel=1,color='fuchsia',secondary_y=False),
                mpf.make_addplot(self.df["MACD_DEM"],panel=1,color='b',secondary_y=True),
                ])
        
        # add 抵扣价
        my_markers = []
        colors = []
        for index in range(len(self.df)):
            marker = None
            color = 'b'
            # 抵扣价使用黄色
            if len(self.df) - 1 - 20 == index or len(self.df)  - 1 - 60 == index or len(self.df) - 1 - 120 == index:
                marker = '*'
                color = 'y'
            my_markers.append(marker)
            colors.append(color)
        added_plots.append(mpf.make_addplot(self.df["Close"], type='scatter', marker=my_markers,markersize=45,color=colors))

        # add bias ratio
        bias_ratio = self.df['bias_ratio']
        added_plots.extend(
            [
                mpf.make_addplot(bias_ratio,panel=3,type='bar',width=0.7,color='b',ylabel="bias_ratio", secondary_y=False),
            ]
        )

        # add rsi
        added_plots.extend([
                mpf.make_addplot(self.df["RSI"], panel=4,color='fuchsia',ylabel="RSI",secondary_y=False),
                mpf.make_addplot([70] * len(self.df), panel=4,color='r', linestyle='--', secondary_y=False),
                mpf.make_addplot([30] * len(self.df), panel=4,color='r', linestyle='--', secondary_y=False),
                mpf.make_addplot([50] * len(self.df), panel=4,color='g', linestyle='--', secondary_y=False),
                ]),

        # add title
        last = len(df) - 1
        daily_percentage = (df["Close"].iloc[last] - df["Close"].iloc[last - 1])/df["Close"].iloc[last - 1] * 100
        daily_percentage = round(daily_percentage, 2)
        
        d1 = df.index[ 0]
        d2 = df.index[-1]
        tdates = [(d1,d2)]

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
        colors = ['r', 'g', 'b', 'y']
        i = 0
        for pivot in result:
            if i == len(result) / 2:
                linestyle="-"
            else:
                linestyle="--"
            axes[0].axhline(y=pivot, color=colors[i % len(colors)], linestyle=linestyle)
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
            fig.savefig(file_name,dpi=300)
            plt.close(fig)
            self.markdown_notes += "\n\n \pagebreak\n\n"
            self.markdown_notes += f"![{image_name}]({image_name}.png)\n\n\n"
            return file_name
        else:
            return fig, axes
    
    def to_markdown(self):
        return self.markdown_notes
    
    def plot_density(self, result_dir=None):
        legend_names = []
        df = self.df
        plt.figure(figsize=(12,9))
        axes = sns.distplot(df["Close"].dropna(), bins=30, color='purple', vertical=True)
        raverage = sum(df["Close"] * df.Volume)/sum(df.Volume)
        raverage = round(raverage, 2)
        today_price = df["Close"].iloc[-1].round(2)
        axes.axhline(y=raverage, color='r', linestyle='--')
        axes.axhline(y=today_price, color='g')
        legend_names.extend(["Volume", f"average={raverage}", f"today={today_price}"])
        axes.legend(legend_names, loc="upper right")
        #step = param.get("step", 5)
        #plt.yticks(np.arange(rmin, rmax, step))
        #plt.grid()
        fig = plt.gcf()
        if result_dir is not None:
            file_name = os.path.join(result_dir, self.name + "_density.png")
            fig.savefig(file_name,dpi=300)
            plt.close(fig)
            self.markdown_notes += f"![{self.name}]({self.name}_density.png)\n\n\n"
            return file_name
        else:
            return axes
    
    def get_pivot(self, **kwargs):

        #pivot_type = "get_large_volume_pivot"
        pivot_type = "get_standard_pivot"
        if 'pivot_type' in kwargs:
            pivot_type = kwargs['pivot_type']
        method = getattr(self, pivot_type)
        return method(**kwargs)


    def get_large_volume_pivot(self, **kwargs):

        interval = 5
        result = []
        indexes = []
        currentMaxLimit = 1.5
        for i in range(interval, len(self.df) - interval):
            currentMax = max(self.df["normalized_volume"].iloc[i - interval:i + interval])
            if currentMax == self.df["normalized_volume"].iloc[i] and currentMax > currentMaxLimit:
                result.append(self.df["High"].iloc[i])
                indexes.append(i)
            
        result_df = self.df.iloc[indexes].round(2)
        result_df = result_df.sort_values(["Close", "normalized_volume"], ascending=False)
        result_df = result_df[["Close", "normalized_volume", "SMA5", "SMA10", "SMA20", "SMA60", "SMA120"]]
        self.markdown_notes += "\n\npivot table: \n\n"
        self.markdown_notes += f"\n\n{result_df.to_markdown()}\n\n"
        return result   

    def get_standard_pivot(self, **kwargs):
        """
        Standard Pivot Points are the most basic Pivot Points. To calculate Standard Pivot Points, you start with a Base Pivot Point, which is the simple average of High, Low and Close from a prior period. A Middle Pivot Point is represented by a line between the support and resistance levels.

        To calculate the Base Pivot Point: (P) = (High + Low + Close)/3
        To calculate the First Support Level: Support 1 (S1) = (P x 2) – High
        To calculate the Second Support Point: Support 2 (S2) = P  –  (High  –  Low)
        To calculate the First Resistance Level: Resistance 1 (R1) = (P x 2) – Low 
        To calculate the Second Resistance Level: Resistance 2 (R2) = P + (High  –  Low) 
        """

        interval = 20
        if 'interval' in kwargs:
            interval = kwargs['interval']
        
        high = max(self.df['High'].iloc[-interval:])
        low = min(self.df['High'].iloc[-interval:])
        close = np.average(self.df["Close"].iloc[-interval:])
        P = (high + low + close) / 3.0
        S1 = P + (P - high)
        S2 = P + (low - high)
        R1 = P + (P - low)
        R2 = P + (high - low)
        result = [S1, S2, P, R1, R2]
        for i in range(len(result)):
            result[i] = round(result[i], 2)
        return result
    
    def get_fibonacci_pivot(self, **kwargs):
        """
        To calculate the Base Pivot Point: Pivot Point (P) = (High + Low + Close)/3 
        To calculate the First Support Level: Support 1 (S1) = P – {.382 * (High  –  Low)} 
        To calculate the Second Support Level: Support 2 (S2) = P – {.618 * (High  –  Low)} 
        To calculate the First Resistance Level: Resistance 1 (R1) = P + {.382 * (High  –  Low)} 
        To calculate the Second Resistance Level: Resistance 2 (R2) = P + {.618 * (High  –  Low)} 
        To calculate the Third Resistance Level: Resistance 3 (R3) = P + {1 * (High  –  Low)} 
        """
        interval = 20
        if 'interval' in kwargs:
            interval = kwargs['interval']
        
        high = max(self.df['High'].iloc[-interval:])
        low = min(self.df['High'].iloc[-interval:])
        close = np.average(self.df["Close"].iloc[-interval:])
        P = (high + low + close) / 3.0
        S1 = P + 0.382 * (low - high)
        S2 = P + 0.618 * (low - high)
        R1 = P + 0.382* (high - low)
        R2 = P + 0.618* (high - low)
        R3 = P + (high - low)
        result = [S1, S2, P, R1, R2, R3]
        for i in range(len(result)):
            result[i] = round(result[i], 2)
        return result

    def calc_average_true_range(self, days=14):
        """
        The Average True Range (ATR) is a tool used in technical analysis to measure volatility
        high = The Current Period High minus
        low = Current Period Low
        prev_close = Previous Close price
        true range = max[(high - low), abs(high - prev_close), abs(low - prev_close)]
        通常情况下股价的波动幅度会保持在一定常态下，
        但是如果有主力资金进出时，股价波幅往往会加剧
        在股价横盘整理、波幅减少到极点时，也往往会产生变盘行情
        根据这个指标来进行预测的原则可以表达为：
        该指标价值越高，趋势改变的可能性就越高;
        该指标的价值越低，趋势的移动性就越弱
        """
        idf = self.df.copy()
        idf['prev_Close'] = idf["Close"]
        for i in range(len(idf)):
            if i == 0:
                idf['prev_Close'].iloc[i] = idf["Close"].iloc[i]
            else: 
                idf['prev_Close'].iloc[i] = idf["Close"].iloc[i-1]
        idf["ATR1"] = abs(idf['High'] - idf['Low'])
        idf["ATR2"] = abs(idf['High'] - idf['prev_Close'])
        idf["ATR3"] = abs(idf['Low'] - idf['prev_Close'])
        idf["ATR"] = idf[["ATR1", "ATR2", "ATR3"]].max(axis=1)
        idf["ATR"] = idf["ATR"].rolling(days).mean()
        self.df["ATR"] = idf["ATR"]
        return self.df["ATR"]

    def calc_relative_strength_index(self, days=14):
        """
        https://www.tradingview.com/ideas/relativestrengthindex/
        """
        idf = self.df.copy()
        self.df["RSI"] = idf.ta.rsi(length=days)
        return self.df["RSI"]


    def calc_momentum_indicator(self, days=14):
        """
        https://www.investopedia.com/articles/technical/081501.asp
        """
        idf = self.df.copy()
        idf['prev_Close'] = idf["Close"]
        for i in range(len(idf)):
            if i-days < 0:
                idf['prev_Close'].iloc[i] = idf["Close"].iloc[i]
            else: 
                idf['prev_Close'].iloc[i] = idf["Close"].iloc[i-days]
        idf['Momentum'] = idf["Close"] - idf['prev_Close']
        self.df["Momentum"] = idf["Momentum"]
        return self.df["Momentum"]

    def calc_william_ratio(self, days=14):
        """
        https://en.wikipedia.org/wiki/Williams_%25R
        """
        idf = self.df.copy()
        high = idf['High'].rolling(days).max()
        low = idf['Low'].rolling(days).min()
        wr =  100 * (idf["Close"] - high) / (high - low)
        wr.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
        self.df['Williams_%R'] = wr
        return self.df['Williams_%R']    

    def calc_money_flow_index(self, days=14):
        """
        https://www.investopedia.com/terms/m/mfi.asp
        """
        idf = self.df.copy()
        typical_price = (idf['High'] + idf['Low'] + idf["Close"])/3.0
        raw_money_flow = typical_price * idf['Volume']
        positive_money_flow = raw_money_flow.copy()
        negative_money_flow = raw_money_flow.copy()
        for i in range(len(idf)):
            if i == 0:
                positive_money_flow.iloc[i] = 0
                negative_money_flow.iloc[i] = 0
            else: 
                prev_close= typical_price.iloc[i-1]
                close = typical_price.iloc[i]
                if close >= prev_close:
                    negative_money_flow.iloc[i] = 0
                else:
                    positive_money_flow.iloc[i] = 0
        
        positive_money_flow = positive_money_flow.rolling(days).mean()
        negative_money_flow  = negative_money_flow.rolling(days).mean()
        raw_money_flow = raw_money_flow.rolling(days).mean()
        mfi = 100 * ( positive_money_flow / raw_money_flow )
        self.df['MFI'] = mfi
        return  self.df['MFI']

    def calc_bias_ratio(self, days=20):
        """
        https://www.investopedia.com/terms/b/bias.asp
        """
        mv = self.df["Close"].rolling(days).mean()
        bias = self.df["Close"] - mv
        self.df["bias_ratio"] = bias
        return self.df['bias_ratio']

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

        apds.extend([
                    mpf.make_addplot(idf["high_trend"],type="line", marker='^', color='r'),
                    mpf.make_addplot(idf["low_trend"], color='r')])
        return apds

    def plot_trend_line(self):
        idf = self.df.copy()
        idf['Date'] = pd.to_datetime(idf.index)
        idf['Date'] = idf['Date'].map(dt.datetime.toordinal)
        data1 = idf.copy()

        rets = np.log(data1["Close"])
        x = data1["Date"]
        slope, intercept, r_value, p_value, std_err = linregress(x, rets)
        idf['Prediction'] = np.e ** (intercept + slope * idf["Date"])

        result = {}
        for days in [365, 2 * 365, 3 * 365]:
            predict_price = np.e ** (intercept + slope * (idf['Date'].iloc[-1] + days) )
            predict_price = round(predict_price, 2)
            result[days] = predict_price
        
        apds = []
        apds.extend([
                    mpf.make_addplot(idf["Prediction"], type="scatter")
                    ])
        return apds, result

    def calc_buy_sell_signal(self):
        idf = self.df.copy()
        buy_score = 0
        sell_score = 0
        messages = []
        #金叉
        if idf["SMA60"].iloc[-1] > idf["SMA120"].iloc[-1] and idf["SMA60"].iloc[-6] < idf["SMA120"].iloc[-6]:
            buy_score += 1
            messages.append("SMA60 cross over SMA120 on up direction")
        if idf["SMA20"].iloc[-1] > idf["SMA60"].iloc[-1] and idf["SMA20"].iloc[-6] < idf["SMA60"].iloc[-6]:
            buy_score += 1
            messages.append("SMA20 cross over SMA60 on up direction")
        if idf["SMA5"].iloc[-1] > idf["SMA20"].iloc[-1] and idf["SMA5"].iloc[-6] < idf["SMA20"].iloc[-6]:
            buy_score += 1
            messages.append("SMA5 cross over SMA20 on up direction")
        
        #死叉
        if idf["SMA60"].iloc[-1] < idf["SMA120"].iloc[-1] and idf["SMA60"].iloc[-6] > idf["SMA120"].iloc[-6]:
            sell_score -= 1
            messages.append("SMA60 cross over SMA120 on down direction")
        if idf["SMA20"].iloc[-1] < idf["SMA60"].iloc[-1] and idf["SMA20"].iloc[-6] > idf["SMA60"].iloc[-6]:
            sell_score -= 1
            messages.append("SMA20 cross over SMA60 on down direction")
        if idf["SMA5"].iloc[-1] < idf["SMA20"].iloc[-1] and idf["SMA5"].iloc[-6] > idf["SMA20"].iloc[-6]:
            sell_score -= 1
            messages.append("SMA5 cross over SMA20 on down direction")

        if idf["MACD_OSC"].iloc[-1] > 0 and idf["MACD_OSC"].iloc[-6] < 0:
            buy_score += 1
            messages.append("MACD cross over 0 on up direction")
        
        if idf["MACD_OSC"].iloc[-1] < 0 and idf["MACD_OSC"].iloc[-6] > 0:
            sell_score -= 1
            messages.append("MACD cross over 0 on down direction")
        
        if idf["RSI"].iloc[-1] > 75:
            sell_score -=1
            messages.append("RSI is too strong, it means overbuy")
        
        if idf["RSI"].iloc[-1] < 25:
            buy_score +=1
            messages.append("RSI is too weak, it means over-sold")
        
        self.attribute["buy_score"] = buy_score
        self.attribute["sell_score"] = sell_score
        self.attribute["buy_sell_comments"] = messages
        return buy_score, sell_score, messages
    
    def get_price_change_table(self):
        """generate result_dict dict.
        key is "5D%, 10D% ..." or moving average "20MA% .."
        value is difference to this baseline
        """
        name_list = ["1D%","5D%", "20D%", "short_term", "mid_term", "long_term", "buy_score"]
        result_dict = {}
        for name in name_list:
            result_dict[name] = self.attribute[name]
        result_dict["Close"] = self.df.Close.iloc[-1]
        result_dict["name"] = self.name
        #result_dict["description"] = self.description
        self.markdown_notes += f"\n\n long_term: {result_dict['long_term']}, "
        self.markdown_notes += f" mid_term: {result_dict['mid_term']}, "
        self.markdown_notes += f" short_term: {result_dict['short_term']}, "
        self.markdown_notes += "\n\n"
        if self.attribute["buy_score"] > 0 or self.attribute["sell_score"] > 0:
            for message in self.attribute["buy_sell_comments"]:
                self.markdown_notes +=  f"- {message}\n"
        self.markdown_notes += "\n\n"

        # bias 
        bias_ratio = self.df["bias_ratio"]
        bias_ratio = bias_ratio.round(2)

        #bias_ratio_min_max_value = max(max(bias_ratio),  abs(min(bias_ratio)))
        #bias_ratio = bias_ratio / bias_ratio_min_max_value
        result_dict["bias_ratio"] = f"{bias_ratio.iloc[-1]}/{bias_ratio.min()}/{bias_ratio.max()}"

        # about volume
        df_volume_mean = self.df['Volume'].mean()
        key_name = "vol_change%"
        num_days = 1
        value = (self.df['Volume'].tail(num_days).sum() - df_volume_mean)/df_volume_mean * 100/num_days
        value = round(value, 2)
        result_dict[key_name] = value
        return result_dict

    def plot_valuation(self):
        stock_name = self.name
        valuation = si.get_stats_valuation(stock_name)
        valuation = valuation.set_index('Unnamed: 0')
        valuation = valuation.applymap(algotrading.stock._convert_to_numeric)
        valuation = valuation.T
        valuation = valuation.sort_index()
        tmp = plt.rcParams["figure.figsize"] 
        plt.rcParams["figure.figsize"] = (20,20)
        valuation[['Trailing P/E','Forward P/E 1', 'PEG Ratio (5 yr expected) 1', 'Price/Sales (ttm)','Price/Book (mrq)']].plot(subplots=True, grid=True)
        plt.rcParams["figure.figsize"] = tmp
        return valuation
    
    def plot_earning(self, **kwargs):

        plt.figure(figsize=(12,9))
        fig, axes = plt.subplots(2)
        
        # add earning history
        earnings_history = si.get_earnings_history(self.name)
        earnings_history = pd.DataFrame(earnings_history)
        earnings_history.dropna(inplace=True)
        earnings_history = earnings_history.set_index("startdatetime")
        earnings_history = earnings_history.sort_index()
        earnings_history.index = pd.to_datetime(earnings_history.index)
        if len(earnings_history) > 10:
            earnings_history = earnings_history.iloc[-10:]
        result_dict = {}
        result_dict["epsactual"] = list(earnings_history["epsactual"])
        result_dict["epsestimate"] = list(earnings_history["epsestimate"])

        earnings = si.get_earnings(self.name)
        info = si.get_analysts_info(self.name)
        result_dict["epsactual"].extend([None, None])
        result_dict["epsestimate"].append(info["Earnings Estimate"].T.iloc[1].loc[1])
        result_dict["epsestimate"].append(info["Earnings Estimate"].T.iloc[2].loc[1])
        result_df = pd.DataFrame(result_dict)
        this_year = dt.datetime.now().year
        next_year = this_year + 1
        new_row = {
            "date": this_year,
            "revenue": _convert_to_numeric(info["Revenue Estimate"].T.iloc[3].loc[1]),
            #"earnings": _convert_to_numeric(info["Earnings Estimate"].T.iloc[3].loc[1])
        }
        earnings["yearly_revenue_earnings"] = earnings["yearly_revenue_earnings"].append(new_row, ignore_index=True)
        new_row = {
            "date": next_year,
            "revenue": _convert_to_numeric(info["Revenue Estimate"].T.iloc[4].loc[1]),
            #"earnings": _convert_to_numeric(info["Earnings Estimate"].T.iloc[4].loc[1])
        }
        earnings["yearly_revenue_earnings"] = earnings["yearly_revenue_earnings"].append(new_row, ignore_index=True)
        earnings["yearly_revenue_earnings"]["revenue"] = earnings["yearly_revenue_earnings"]["revenue"]/1000000000
        #earnings["yearly_revenue_earnings"] = earnings["yearly_revenue_earnings"]/1000000000
        earnings["yearly_revenue_earnings"].set_index('date', inplace=True)
        earnings["yearly_revenue_earnings"]["revenue"].plot(ax=axes[0], marker='o', legend=["revenue(B)"])
        #print(result_df)
        result_df.plot(ax=axes[1], marker='o')
 
        # save to file if possible
        #fig = plt.gcf()
        image_name = self.name
        if "image_name" in kwargs:
            image_name = kwargs['image_name']
        result_dir = kwargs['result_dir'] if 'result_dir' in kwargs else None
        if result_dir is not None:
            file_name =  image_name + "_earnings.png"
            fig.savefig(os.path.join(result_dir,file_name),dpi=300)
            plt.close(fig)
            self.markdown_notes += "\n\n \pagebreak\n\n"
            self.markdown_notes += f"![{image_name}]({file_name})\n\n\n"
            return file_name
        else:
            return fig, axes

 
    def add_quick_summary(self):
        quick_summary_md = ""
        quick_summary_md += f"# Quick summary about {self.name}\n\n"
        quick_summary_md += "This is the quick summary:\n\n"

        try:
            status, messages, _ = self.is_good_business()
            quick_summary_md += f"{self.name} is a good business? \nAnswer: {status}\n\n"
            if status == False:
                quick_summary_md += f"Reason:, {messages}\n\n"
        except:
            pass

        for item in [
            "sector",
            "longBusinessSummary",  "country", "city", "trailingPE",  "priceToSalesTrailing12Months", "fiftyTwoWeekHigh","fiftyTwoWeekLow",
            "pegRatio", "shortPercentOfFloat", "next_earnings_date"
            ]:
            try:
                tmp_str = f"- {item}: {self.attribute[item]}\n\n"
                quick_summary_md += tmp_str
            except:
                pass
        
        info = si.get_stats_valuation(self.name)
        quick_summary_md += info.to_markdown() + "\n\n"

        info = si.get_analysts_info(self.name)
        quick_summary_md += info["Growth Estimates"].to_markdown() + "\n\n"
        #quick_summary_md += info["Revenue Estimate"].to_markdown() + "\n\n"


        # finally, add it into markdown_notes  
        self.markdown_notes += quick_summary_md
        return quick_summary_md


    def add_notes(self):
        notes = algotrading.data.get_notes(self.name)
        self.markdown_notes += notes
        return notes


 