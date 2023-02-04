import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import algotrading

# Load the stock data into a pandas DataFrame
name = "SQQQ"
stock = algotrading.stock.Stock(name)
stock.read_data()
df = stock.df

# Calculate the moving average convergence divergence (MACD)
def macd(close, fast_window=12, slow_window=26, signal_window=9):
    ema_fast = close.ewm(span=fast_window).mean()
    ema_slow = close.ewm(span=slow_window).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window).mean()
    histogram = macd - signal
    return macd, signal, histogram


macd, signal, histogram = macd(df["Close"])
df["MACD"] = macd
df["Signal Line"] = signal

# Plot the stock prices and MACD in separate sub-panels
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(df["Close"], label="Close", color="blue")
axs[0].set_xlabel("Time")
axs[0].set_title(f"{name} at {dt.datetime.now()}")
axs[0].set_ylabel("Close", color="blue")

axs[1].plot(df["MACD"], label="MACD", color="red")
axs[1].plot(df["Signal Line"], label="Signal Line", color="green")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("MACD/Signal Line", color="red")

plt.legend(loc="best")
plt.show()
