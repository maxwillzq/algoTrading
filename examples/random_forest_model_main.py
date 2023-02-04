import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import algotrading

# Load the stock data into a pandas DataFrame
name = "SQQQ"
stock = algotrading.stock.Stock(name)
stock.read_data(days=1000)
df = stock.df

# Extract the close prices
data = df[["Close"]]

# Add lagged values as features
fill_value = data["Close"].iloc[0]
for i in range(1, 31):
    data["lag_{}".format(i)] = data["Close"].shift(i, fill_value=fill_value)

# Remove the rows with NaN values
data = data.dropna()

# Extract the features and target variables
features = data.drop("Close", axis=1)
target = data["Close"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train a Random Forest Regressor on the training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the stock prices using the trained model
y_pred = model.predict(X_test)

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error: ", rmse)

# Predict the next 30 days of stock prices
future_dates = pd.date_range(data.index[-1], periods=30, freq="D")
future_features = pd.DataFrame(index=future_dates[1:], columns=data.columns)
future_data = pd.concat([data, future_features])

# Add lagged values to the future data
fill_value = data["Close"].iloc[-1]
for i in range(1, 31):
    future_data["lag_{}".format(i)] = future_data["Close"].shift(
        i, fill_value=fill_value
    )

X_future = future_data.drop("Close", axis=1).iloc[-30:]
X_future.fillna(fill_value, inplace=True)
try:
    y_future_pred = model.predict(X_future)
except:
    raise ValueError(f"input is wrong X_future = {type(X_future)}")

# Plot the prediction results for the next 30 days along with the old data
plt.figure(figsize=(12, 8))
# plt.plot(data.index, data["Close"], label="Old Data")
plt.plot(future_dates, y_future_pred, label="Prediction")
plt.xlabel("Date")
plt.ylabel
plt.show()
