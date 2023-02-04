import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import algotrading

# Load the stock data into a pandas DataFrame
name = "BABA"
stock = algotrading.stock.Stock(name)
stock.read_data()
df = stock.df

# Calculate the 20-day moving average and standard deviation
df["20-day MA"] = df["Close"].rolling(window=20).mean()
df["20-day STD"] = df["Close"].rolling(window=20).std()

# Calculate the upper and lower Bollinger Bands
df["Upper Band"] = df["20-day MA"] + 2 * df["20-day STD"]
df["Lower Band"] = df["20-day MA"] - 2 * df["20-day STD"]

# Plot the stock prices and Bollinger Bands
plt.plot(df["Close"], label="Close")
plt.plot(df["Upper Band"], label="Upper Band", color="red")
plt.plot(df["Lower Band"], label="Lower Band", color="green")
plt.legend()
plt.title(f"{name} at {dt.datetime.now()}")
plt.show()
