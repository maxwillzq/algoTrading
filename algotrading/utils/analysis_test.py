from __future__ import absolute_import, division, print_function, unicode_literals

from absl.testing import absltest
import datetime as dt
import pandas as pd
import yfinance as yf
import tempfile

import algotrading
from algotrading.utils import analysis
from algotrading.utils import test_utils


class TestPlotting(test_utils.TestCaseBase):

    def test_ma_discount(self):
        df = self.stock.df
        discount = analysis.ma_discount(df)
        # discount.plot(title="ma_discount")
        # import matplotlib.pyplot as plt
        # plt.show()
        self.assertIsNotNone(discount)

    def test_calc_volatility(self):
        analysis.calc_volatility(["GOOG", "AAPL", "AMZN"])


if __name__ == "__main__":
    absltest.main()
