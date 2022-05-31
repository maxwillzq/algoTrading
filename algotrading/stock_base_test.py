from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
import os

import algotrading
from algotrading.utils import plotting
from algotrading import stock_base

TOP_DIR = os.path.dirname(algotrading.__file__)


class TestStockBaseClass(unittest.TestCase):
    """Test stock_base class."""
    def setUp(self) -> None:
        """Create StockBase object and download all data.
        """
        super().setUp()
        self.stock = stock_base.StockBase('AMZN')
        self.stock.read_data(data_input_file=os.path.join(TOP_DIR, "test_data/AMZN_20220530.csv"))
        self.stock.generate_more_data()
    
    def test_generate_more_data(self):
        # test calc_moving_average func
        for item in ['SMA5', 'SMA10', 'SMA20', 'SMA60', 'SMA120', 'SMA240']:
            self.assertTrue(item in self.df)
        for item in ['EMA5', 'EMA10', 'EMA20', 'EMA60', 'EMA120', 'EMA240']:
            self.assertTrue(item in self.df)
        # test calc_macd
        for item in ['MACD_DIF', 'MACD_DEM', 'MACD_OSC']:
            self.assertTrue(item in self.df)


    def test_is_good_business(self) -> None:
        result = self.stock.is_good_business()
        self.assertFalse(result[0])

    def test_plot_valuation(self):
        self.stock.plot_valuation()
    
    def test_test(self):
        pass
    
if __name__ == "__main__":
    unittest.main()
