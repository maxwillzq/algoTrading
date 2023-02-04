import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import algotrading

# Load the stock data into a pandas DataFrame
name = "BABA"
stock = algotrading.stock.Stock(name)
stock.read_data(1000)
df = stock.df

# Calculate short-term and long-term moving averages
short_rolling = df.rolling(window=20, min_periods=1).mean()
long_rolling = df.rolling(window=100, min_periods=1).mean()

# Create a new dataframe to store the buy/sell signals
signals = pd.DataFrame(index=df.index)
signals["signal"] = 0.0

# Generate the signals
signals["signal"][20:] = np.where(
    short_rolling["Close"][20:] > long_rolling["Close"][20:], 1.0, 0.0
)
signals["positions"] = signals["signal"].diff()

# Plot the original data and the signals
fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(111, ylabel="Price in $")
df["Close"].plot(ax=ax1, color="black", lw=2.0)
short_rolling["Close"].plot(ax=ax1, label="20 days rolling mean", color="red")
long_rolling["Close"].plot(ax=ax1, label="100 days rolling mean", color="green")

# Plot the buy/sell signals
ax1.plot(
    signals.loc[signals.positions == 1.0].index,
    df.Close[signals.positions == 1.0],
    "^",
    markersize=10,
    color="g",
)
ax1.plot(
    signals.loc[signals.positions == -1.0].index,
    df.Close[signals.positions == -1.0],
    "v",
    markersize=10,
    color="r",
)

plt.legend(loc="upper left")
plt.show()
