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
from typing import Optional,Mapping
logger = logging.getLogger(__name__)

def plot_price_volume(df: pd.DataFrame, stock_name: Optional[str]=None, param: Optional[Mapping]={}):
  """
    This function creates a subplot of two graphs, the first graph shows the "Adj Close"
    value of the stock, and the second graph shows the stock's trading "Volume".
    Additionally, this function can also plot the linear regression trendline of the "Adj Close" 
    values for the specified data range in the "data_range_list" parameter. 

    Parameters:
    df (pandas.DataFrame): The dataframe containing the stock data. 
    stock_name (str, optional): The name of the stock to be displayed on the title of the plot. 
                                If not provided, the title will be empty.
    param (dict, optional): A dictionary of optional parameters. 
                            The following parameter is supported:
                            - data_range_list (list): A list of integers, representing the data range
                              in days to be used to calculate the linear regression trendline. 
                              The default is [90].

    Returns:
    None

    """
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
    x_ind = range(len(rets))
    slope, intercept, r_value, p_value, std_err = linregress(x_ind, rets)
    print("Linear" + str(data_range) + " = " + str(slope))
    top.plot(x, np.e ** (intercept + slope*x_ind), label="Linear" + str(data_range))
  legend = top.legend(loc='upper left', shadow=True, fontsize='x-large')
  plt.show()

def plot_price_density(df: pd.DataFrame, param: Optional[Mapping]={}):
  """
  Plots a density plot of the stock's adjusted close prices.

  Args:
      df (pandas.DataFrame): The dataframe containing the stock data.
      param (dict, optional): Additional parameters to customize the plot. Defaults to an empty dictionary.

  Returns:
      None

  Raises:
      None
  """
  plt.figure(figsize=(12,9))
  ax = sns.distplot(df['Adj Close'].dropna(), bins=50, color='purple', vertical=True)
  rmin = min(df['Adj Close']) * 0.9
  rmax = max(df['Adj Close']) * 1.1
  step = param.get("step", 5)
  plt.yticks(np.arange(rmin, rmax, step))
  plt.grid()

def plot_moving_average(df: pd.DataFrame, stock_name: str, param: Mapping={'list': [20, 60, 120]}, file_name: Optional[str]=None):
  """
  Plots the stock's prices along with moving averages.

  Args:
      df (pandas.DataFrame): The dataframe containing the stock data.
      stock_name (str): The name of the stock to be plotted.
      param (dict, optional): Additional parameters to customize the plot. Defaults to an empty dictionary.
      file_name (str, optional): The file name to save the plot. Defaults to None.

  Returns:
      None

  Raises:
      None
  """
  # simple moving averages
  lists = param["list"]
  if file_name:
    mplfinance.plot(df, 
      type='candle', 
      mav=lists, 
      volume=True,
      figsize=(12, 9), 
      title=stock_name,
      savefig=file_name
      )
  else:
    mplfinance.plot(df, 
      type='candle', 
      mav=lists, 
      volume=True,
      figsize=(12, 9), 
      title=stock_name)

def plot_price_minus_moving_average(df: pd.DataFrame, stock_name: str, param: Mapping={'MA': 50}):
  """Plots the difference between stock's adjusted close price and its moving average.

  Args:
      df (pd.DataFrame): The dataframe containing the stock data.
      stock_name (str): The name of the stock.
      param (dict, optional): Additional parameters to customize the plot. Defaults to an empty dictionary.

  Returns:
      None

  Raises:
      None
  """
  ma = param["MA"]
  df[f'MA{ma}'] = df['Adj Close'].rolling(ma).mean()
  plt.plot(df.index, df['Adj Close'] - df[f'MA{ma}'])
  plt.title(f"{stock_name} Price - MA{ma}" )
  plt.grid()
  plt.show()

def ma_discount(df, param={'MA': 50}):
  """
  The discount rate using moving average as baseline.
  """
  MA = param['MA']
  result =  -(df['Adj Close'] - df['Adj Close'].rolling(MA).mean())/df['Adj Close'] * 100
  return result

def linear_regression_gains(df,param={'MA': 90}):
  """
  The potential gain rate after 1 year use linear model
  """

  def momentum(closes):
      returns = np.log(closes)
      x = np.arange(len(returns))
      slope, _, rvalue, _, _ = linregress(x, returns)
      # annualize slope and multiply by R^2
      return ((1 + slope) ** 252) * (rvalue ** 2)  
  
  
  MA = param['MA']
  result = (df.rolling(MA)['Adj Close'].apply(momentum, raw=False) - 1)
  return result

def calc_volatility(stock_name_list, output_file_name=None):
    """
    Calculates the volatility of a set of stock prices and plots the results as a bar graph.
    https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
    """
    stocks = [algotrading.stock.Stock(name) for name in stock_name_list]
    for stock in stocks:
      stock.read_data()
    data = {stock.name: stock.df['Close'] for stock in stocks}
    data = pd.DataFrame(data)
    returns = data.pct_change().apply(lambda x: np.log(1 + x))
    volatility = returns.std().apply(lambda x: x * np.sqrt(250))
    volatility.sort_values(ascending=True, inplace=True)
    average = volatility.mean().round(2)
    
    fig, ax = plt.subplots()
    volatility.plot(kind='bar', ax=ax)
    ax.axhline(y=average, color='y', linestyle='--',label="volatility")
    ax.set_title("volatility index")
    ax.legend()

    if output_file_name:
        fig.savefig(output_file_name,dpi=300)
        plt.close(fig)
        return output_file_name
    else:
        return ax

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
