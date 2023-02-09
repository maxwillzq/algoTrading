import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download data for the stock with ticker symbol 'AAPL'
data = yf.download('SQQQ')

# Calculate the daily returns
data['Returns'] = data['Close'].pct_change()

# Calculate the standard deviation of daily returns
volatility = data['Returns'].std() * (252 ** 0.5)

# Plot the stock's daily returns and volatility
plt.figure(figsize=(12, 8))
plt.hist(data['Returns'], bins=100, density=True, alpha=0.7, label='Returns')
plt.axvline(volatility, color='r', linestyle='dashed', linewidth=2, label='Volatility')
plt.xlabel('Return')
plt.ylabel('Density')
plt.title('Apple Inc. (AAPL) Stock Returns and Volatility')
plt.legend(loc='best')
plt.show()
