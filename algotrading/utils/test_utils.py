from __future__ import absolute_import, division, print_function, unicode_literals

from absl.testing import absltest
import os
import pandas as pd

import algotrading
from algotrading.utils import plotting
from algotrading import stock_base

TOP_DIR = os.path.dirname(algotrading.__file__)


class TestCaseBase(absltest.TestCase):
    """Test stock_base class."""

    def setUp(self) -> None:
        """Create StockBase object and download all data."""
        super().setUp()
        self.stock = stock_base.StockBase("AMZN")
        self.stock.read_data(
            data_input_file=os.path.join(TOP_DIR, "test_data/AMZN_20220530.csv")
        )
        if "Adj Close" not in self.stock.df.columns:
            self.stock.df["Adj Close"] = self.stock.df["Close"]
        self.stock.df.index = pd.to_datetime(self.stock.df["Date"])
