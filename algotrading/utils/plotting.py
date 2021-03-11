import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import datetime as dt
import mplfinance
import logging
import numpy as np
logger = logging.getLogger(__name__)

def draw_regular_plot(df, stock_name=None, param={}):
  plt.figure(figsize=(12,9))
  top = plt.subplot2grid((12,9), (0, 0), rowspan=10, colspan=9)
  bottom = plt.subplot2grid((12,9), (10,0), rowspan=2, colspan=9)
  top.plot(df.index, df['Adj Close'], color='blue') #df.index gives the dates
  top.grid()
  bottom.bar(df.index, df['Volume'])
  # set the labels
  top.axes.get_xaxis().set_visible(False)
  if stock_name:
    top.set_title(stock_name)
  top.set_ylabel('Adj Close')
  bottom.set_ylabel('Volume')
  last = len(df)
  data_range_list = [90]
  if 'data_range_list' in param:
    data_range_list = param["data_range_list"]
  for data_range in data_range_list:
    rets = np.log(df['Adj Close'].iloc[last - data_range : last])
    x = df.index[last-data_range:last]
    slope, intercept, r_value, p_value, std_err = linregress(x, rets)
    print("Linear" + str(data_range) + " = " + str(slope))
    top.plot(x, np.e ** (intercept + slope*x), label="Linear" + str(data_range))
  legend = top.legend(loc='upper left', shadow=True, fontsize='x-large')
  plt.show()

def draw_density_plot(df, param = {}):
  plt.figure(figsize=(12,9))
  ax = sns.distplot(df['Adj Close'].dropna(), bins=50, color='purple', vertical=True)
  rmin = min(df['Adj Close']) * 0.9
  rmax = max(df['Adj Close']) * 1.1
  step = param.get("step", 5)
  plt.yticks(np.arange(rmin, rmax, step))
  plt.grid()

def draw_moving_average_plot(df, stock_name, param={}, file_name=None):
  # simple moving averages
  lists = [20, 60, 120]
  if "list" in param:
    lists = param["list"]
  plot = mplfinance.plot(df, 
    type='candle', 
    mav=lists, 
    volume=True,
    figsize=(12, 9), 
    title=stock_name,
    savefig=file_name
    )

def draw_price_to_ma_distance(df, stock_name, param={}):
  ma = 50
  if "MA" in param:
    ma = param["MA"]
  df['MA' + str(ma)] = df['Adj Close'].rolling(ma).mean()
  plt.plot(df.index, df['Adj Close'] - df['MA' + str(ma)])
  plt.title(f"{stock_name} Price - MA{ma}" )
  plt.grid()
  plt.show()

def indicator_1(df, param={}):
  """
  The discount rate if use MA price as baseline
  """
  MA = 50
  if 'MA' in param:
    MA = param['MA']
  result =  -(df['Adj Close'] - df['Adj Close'].rolling(MA).mean())/df['Adj Close'] * 100
  return result

def indicator_2(df,param={}):
  """
  The potential gain rate after 1 year use linear model
  """

  def momentum(closes):
      returns = np.log(closes)
      x = np.arange(len(returns))
      slope, _, rvalue, _, _ = linregress(x, returns)
      return ((1 + slope) ** 252) * (rvalue ** 2)  # annualize slope and multiply by R^2
  
  MA = 90
  if 'MA' in param:
    MA = param['MA']
  result = (df.rolling(MA)['Adj Close'].apply(momentum, raw=False) - 1)
  #print("result = ", result)
  return result