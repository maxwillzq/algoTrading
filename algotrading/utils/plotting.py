import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import datetime as dt
import mplfinance
import logging
import numpy as np
import os
import algotrading
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

def calc_volatility(stock_name_list, output_file_name=None):
    """
    https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
    """
    test = {}
    for stock_name in stock_name_list:
        stock = algotrading.stock.Stock(stock_name)
        stock.read_data()
        test[stock_name] = stock.df['Close']
    test = pd.DataFrame(test)
    volatility_df = test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    volatility_df = volatility_df.sort_values(ascending=True)
    average = np.mean(volatility_df).round(2)
    axes = volatility_df.plot(kind='bar')
    axes.axhline(y=average, color='y', linestyle='--',label="volatility")
    axes.set_title("volatility index")

    fig = plt.gcf()
    if output_file_name:
        fig.savefig(output_file_name,dpi=300)
        plt.close(fig)
        return output_file_name
    else:
        return axes

def generate_portfolio(stock_name_list, result_dir=None):
    """
    https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
    """
    test = {}
    for stock_name in stock_name_list:
        stock = algotrading.stock.Stock(stock_name) 
        end = end = dt.datetime.now()
        start = start =  end - dt.timedelta(3000)
        stock.read_data(start, end)
        test[stock_name] = stock.df['Close']
    test = pd.DataFrame(test)
    markdown_notes = "portfolio analysis \n\n\n\n"

    # Yearly returns for individual companies
    ind_er = test.resample('Y').last().pct_change().mean()
    markdown_notes += ind_er.to_markdown()
    markdown_notes += "\n\n"
    cov_matrix = test.pct_change().apply(lambda x: np.log(1+x)).cov()
    corr_matrix = test.pct_change().apply(lambda x: np.log(1+x)).corr()
    markdown_notes += "corr matrix \n\n"
    markdown_notes += corr_matrix.to_markdown()
    markdown_notes += "\n\n"

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_assets = len(test.columns)
    num_portfolios = 10000

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) 
        
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd)
    
    data = {'Returns':p_ret, 'Volatility':p_vol}

    for counter, symbol in enumerate(test.columns.tolist()):
        #print(counter, symbol)
        data[symbol+' weight'] = [w[counter] for w in p_weights]
    portfolios  = pd.DataFrame(data)

    # the minimum volatility portfolio
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]

    # Finding the optimal portfolio
    rf = 0.01 # risk factor
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]

    markdown_notes += "optimal risk protfolio: \n\n"
    markdown_notes += optimal_risky_port.to_markdown()
    markdown_notes += "\n\n"

    # Plotting optimal portfolio
    plt.subplots(figsize=(10, 10))
    plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
    fig = plt.gcf()
    if result_dir:
        fig.savefig(os.path.join(result_dir, "protfolio.png"),dpi=300)
        plt.close(fig)
        with open(os.path.join(result_dir, "protfolio.md"), 'w') as f:
            f.write(markdown_notes)
    else:
        plt.show()
        return fig, markdown_notes
