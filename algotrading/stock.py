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
    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        self.markdown_notes = "\n\\pagebreak\n\n"
        title = self.description if self.description else self.name
        self.markdown_notes += f"## {title}\n\n"


    def read_data(self, start = None, end = None):
        """Read Stock Data from Yahoo Finance
        """
        if end is None:
            end = dt.datetime.now()
        if start is None:
            start =  end - dt.timedelta(365)
        logger.info(f"today is {end}")
        if not start:
            start = dt.datetime(end.year-2, end.month, end.day)
        try:
            df = web.DataReader(self.name, 'yahoo', start, end)
        except:
            logger.error(f"fail to get data for symbol {self.name}")
            raise RuntimeError()
        df.index = pd.to_datetime(df.index)
        round_df = df.round(2)
        self.markdown_notes += f"{round_df.tail(5).to_markdown()}\n"
        self.df = df
        return self.df
    
    def get_ma_range_min_max(self, MA=20):
        last = len(self.df)
        mav = self.df['Close'].rolling(MA).mean()
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
        idf['60_EMA'] = idf['Close'].rolling(60).mean()
        idf['120_EMA'] = idf['Close'].ewm(span=120, adjust=False).mean()
        idf['Signal'] = 0.0  
        idf['Signal'] = np.where(macd > signal + 0.02, 1.0, 0.0)
        idf['Position'] = idf['Signal'].diff()
        bias_120 = (idf['Close'] - idf['120_EMA']) / idf['120_EMA'] * 100
        apds.extend(
            [
                mpf.make_addplot(bias_120,panel=3,type='bar',width=0.7,color='b',ylabel="bias_120", secondary_y=False),
            ]
        )

        my_markers = []
        colors = []
        index = 0
        for i, v in idf['Position'].items():
            marker = None
            color = 'b'
            if v == 1 and idf.loc[i]["Close"] <= max(idf.loc[i]["60_EMA"], idf.loc[i]["20_EMA"]) * 1.05:
                # Buy point
                #marker = '^'
                #color = 'g'
                #logger.debug(f"index = {i}, macd = {macd.loc[i]}, signal = {signal.loc[i]}, hist = {histogram.loc[i]}")
                pass
            elif v == -1 and idf.loc[i]["Close"] >= max(idf.loc[i]["20_EMA"],idf.loc[i]["60_EMA"]):
                # Sell point
                # marker = 'v'
                marker = None
                color = 'r'
            # 抵扣价使用黄色
            if len(idf) - 1 - 20 == index or len(idf) - 1 - 60 == index or len(idf) - 1 - 120 == index:
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
        result_dict['name'] = self.name
        result_dict['description'] = self.description
        last = len(self.df) - 1
        for delta in [1, 5, 20, 60, 120, 240]:
            key_name = f"{delta}D%"
            if last - delta > 0:
                value = (self.df['Close'].iloc[last] - self.df['Close'].iloc[last - delta])/self.df['Close'].iloc[last - delta] * 100
                value = round(value, 2)
                result_dict[key_name] = value
            else:
                result_dict[key_name] = None

        """
        for delta in [20, 60, 120]:
            key_name = f"bias_{delta}MA"
            df_EMA = self.df['Close'].rolling(delta).mean().round(2)
            value = (self.df['Close'].iloc[last] - df_EMA.iloc[last])/df_EMA.iloc[last] * 100
            value = round(value, 2)
            result_dict[key_name] = value
        """

        # about volume
        df_volume_EMA = self.df['Volume'].rolling(20).mean().round(2)
        key_name = "vol_change%"
        value = (self.df['Volume'].iloc[last] - df_volume_EMA.iloc[last])/df_volume_EMA.iloc[last] * 100
        value = round(value, 2)
        result_dict[key_name] = value
        
        return result_dict
    
    def plot(self, result_dir=None, apds=[]):
        
        df = self.df.copy()
        mav = [20, 60, 120]
        colors = ['r', 'g', 'b', 'y']
        legend_names = []
        added_plots = []
        for index in range(len(mav)):
            item = mav[index]
            color = colors[index]
            if len(df) <= item:
                break
            df_sma = df['Close'].rolling(item).mean()
            df_ema = df['Close'].ewm(span=item, adjust=False).mean()
            added_plots.append(
                mpf.make_addplot(df_sma, color=color)
            )
            legend_names.append(f"SMA{item}")
            added_plots.append(
                mpf.make_addplot(df_ema, color=color, linestyle='--')
            )
            legend_names.append(f"EMA{item}")
        added_plots.extend(apds)
        last = len(df) - 1
        delta = 1
        daily_percentage = (df['Close'].iloc[last] - df['Close'].iloc[last - delta])/df['Close'].iloc[last - delta] * 100
        daily_percentage = round(daily_percentage, 2)
        if len(added_plots) > 0:
            fig, axes = mpf.plot(df, 
                type='candle', 
                style="yahoo",
                volume=True,
                figsize=(12, 9), 
                title=f"Today's increase={daily_percentage}%",
                returnfig=True,
                volume_panel=2,
                addplot=added_plots,
                )
        else:
            fig, axes = mpf.plot(df, 
                type='candle', 
                style="yahoo",
                volume=True,
                figsize=(12, 9), 
                title=f"Today's increase={daily_percentage}%",
                returnfig=True,
                volume_panel=1,
                )

        # Configure chart legend and title
        rmin, rmax = self.get_ma_range_min_max(MA=60)
        rmin = round(rmin, 2)
        rmax = round(rmax, 2)
        axes[0].axhline(y=rmin, color='y', linestyle='--')
        axes[0].axhline(y=rmax, color='y', linestyle='--')
        legend_names.extend([rmin, rmax])

        axes[0].legend(legend_names, loc="upper left")
        if result_dir is not None:
            file_name = os.path.join(result_dir, self.name + ".png")
            fig.savefig(file_name,dpi=300)
            plt.close(fig)
            self.markdown_notes += f"![{self.name}]({self.name}.png)\n\n\n"
            return file_name
        else:
            return fig
    
    def to_markdown(self):
        return self.markdown_notes
    
    def plot_density(self, result_dir=None):
        legend_names = []
        df = self.df
        plt.figure(figsize=(12,9))
        axes = sns.distplot(df['Adj Close'].dropna(), bins=30, color='purple', vertical=True)
        raverage = sum(df['Adj Close'] * df.Volume)/sum(df.Volume)
        raverage = round(raverage, 2)
        today_price = df['Adj Close'].iloc[-1].round(2)
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
            return fig

class Fred:
    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        self.markdown_notes = "\n\\pagebreak\n\n"
        title = self.description if self.description else self.name
        self.markdown_notes += f"## {title}\n\n"

    def read_data(self, start = None, end = None):
        """Read Stock Data from Yahoo Finance
        """
        if end is None:
            end = dt.datetime.now()
        logger.info(f"today is {end}")
        if not start:
            start = dt.datetime(end.year-2, end.month, end.day)
        
        df = web.DataReader(self.name, 'fred', start, end)
        df.index = pd.to_datetime(df.index)
        round_df = df.round(2)
        self.markdown_notes += f"{round_df.tail(5).to_markdown()}\n"
        self.df = df
        return self.df

    def plot(self, result_dir=None, apds=[]):
        mav = [20, 60, 120, 200, 300]
        legend_names = [f"MA{item}" for item in mav]
        last = len(self.df) - 1
        delta = 1
        daily_percentage = (self.df[self.name].iloc[last] - self.df[self.name].iloc[last - delta])/self.df[self.name].iloc[last - delta] * 100
        daily_percentage = round(daily_percentage, 2)
        plt.figure(figsize=(12,9))
        df = self.df.copy()
        df["MA20"] = df[self.name].rolling(20).mean()
        #df["MA20"].plot()
        df.plot(y=[self.name])
        plt.legend(loc='upper left', shadow=True, fontsize='x-large')
        plt.grid()
        plt.title(f"Today's increase={daily_percentage}%")

        if result_dir is not None:
            file_name = os.path.join(result_dir, self.name + ".png")
            plt.savefig(file_name,dpi=300)
            plt.close()
            return file_name
        else:
            fig = plt.gcf()
            return fig
    
    def get_price_change_table(self):
        """generate result_dict dict.
        key is "5D%, 10D% ..." or moving average "20MA% .."
        value is difference to this baseline
        """
        result_dict = {}
        result_dict["name"] = self.name
        if self.description:
            result_dict["description"] = self.description
        last = len(self.df) - 1
        for delta in [1, 5, 20, 60, 120, 240]:
            key_name = f"{delta}D%"
            if last - delta > 0:
                value = (self.df[self.name].iloc[last] - self.df[self.name].iloc[last - delta])/self.df[self.name].iloc[last - delta] * 100
                value = round(value, 2)
                result_dict[key_name] = value
            else:
                result_dict[key_name] = None
        
        return result_dict

    def to_markdown(self):
        return self.markdown_notes