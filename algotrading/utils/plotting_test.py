from __future__ import absolute_import, division, print_function, unicode_literals

from absl.testing import absltest
import datetime as dt
import pandas as pd
import yfinance as yf
import tempfile

import algotrading
from algotrading.utils import plotting
from algotrading.utils import test_utils


class TestPlotting(test_utils.TestCaseBase):
    def test_plot_price_volume(self):
        df = self.stock.df
        plotting.plot_price_volume(df)

    def test_plot_price_density(self):
        df = self.stock.df
        plotting.plot_price_density(df)

    def test_plot_moving_average(self):
        df = self.stock.df
        plot = plotting.plot_moving_average(df, stock_name="AMZN")

    def test_plot_price_minus_moving_average(self):
        df = self.stock.df
        plot = plotting.plot_price_minus_moving_average(df, stock_name="AMZN")

    def test_ma_discount(self):
        df = self.stock.df
        discount = plotting.ma_discount(df)
        # discount.plot(title="ma_discount")
        # import matplotlib.pyplot as plt
        # plt.show()
        self.assertIsNotNone(discount)

    def test_calc_volatility(self):
        plotting.calc_volatility(["GOOG", "AAPL", "AMZN"])

    def test_data_reader(self):
        # Define the ticker symbol and date range for the stock data
        ticker = "AAPL"
        today = dt.datetime.now()
        start_date = today - dt.timedelta(3000)
        end_date = today
        df = yf.download(ticker, start=start_date, end=end_date)
        self.assertFalse(df.empty)
        # Assert that the resulting DataFrame has the expected columns
        expected_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        self.assertEqual(list(df.columns), expected_columns)
        # Assert that the resulting DataFrame has the expected number of rows
        expected_rows = 2066
        self.assertEqual(len(df), expected_rows)


if __name__ == "__main__":
    absltest.main()
