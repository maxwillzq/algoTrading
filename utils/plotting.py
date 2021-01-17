import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import logging
import numpy as np
logger = logging.getLogger(__name__)

def draw_regular_plot(df, stock_name=None, param={}):
  plt.figure(figsize=(12,9))
  top = plt.subplot2grid((12,9), (0, 0), rowspan=10, colspan=9)
  bottom = plt.subplot2grid((12,9), (10,0), rowspan=2, colspan=9)
  top.plot(df.index, df['Adj Close'], color='blue') #df.index gives the dates
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

def draw_density_plot(df, param = {}):
  plt.figure(figsize=(12,9))
  ax = sns.distplot(df['Adj Close'].dropna(), bins=50, color='purple', vertical=True)
  rmin = min(df['Adj Close']) * 0.9
  rmax = max(df['Adj Close']) * 1.1
  step = param.get("step", 5)
  plt.yticks(np.arange(rmin, rmax, step))

def draw_moving_average_plot(df, param={}):
  # simple moving averages
  lists = [20, 50, 200]
  if "list" in param:
    lists = param["list"]
  args = {}
  for item in lists:
    df['MA' + str(item)] = df['Adj Close'].rolling(item).mean()
    args['MA' + str(item)] = df['MA' + str(item)]
  args['Adj Close'] = df['Adj Close']
  df2 = pd.DataFrame(args)
  df2.plot(figsize=(12, 9), legend=True, title=stock_name)
  #df2.to_csv('AAPL_MA.csv')
  fig = plt.gcf()
  fig.set_size_inches(12, 9)
  #fig.savefig('AAPL_plot.png', dpi=300)
  plt.show()
  print(df.tail())

def draw_price_to_ma_distance(df, stock_name, param={}):
  ma = 50
  if "MA" in param:
    ma = param["MA"]
  df['MA' + str(ma)] = df['Adj Close'].rolling(ma).mean()
  plt.plot(df.index, df['Adj Close'] - df['MA' + str(ma)])
  plt.title(f"{stock_name} Price - MA{ma}" )
  plt.show()