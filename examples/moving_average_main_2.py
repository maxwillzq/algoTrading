import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import algotrading

# Load the stock data into a pandas DataFrame
name = "SHOP"
stock = algotrading.stock.Stock(name)
stock.read_data(days=3000)
df = stock.df

# Calculate the short and long moving averages
short_rolling = df["Close"].rolling(window=50).mean()
long_rolling = df["Close"].rolling(window=200).mean()

# Plot the stock price and moving averages
fig, ax = plt.subplots()
ax.plot(df["Close"], label="Stock Price")
ax.plot(short_rolling, label="50-day Moving Average")
ax.plot(long_rolling, label="200-day Moving Average")
ax.legend(loc="best")
ax.set_title(f"{name} at {dt.datetime.now()}")
plt.show()
