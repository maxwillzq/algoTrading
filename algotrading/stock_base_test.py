from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from absl.testing import absltest
import os
import algotrading
from algotrading.utils import plotting
from algotrading.utils import test_utils
from algotrading import stock_base

 class TestConvertToNumeric(absltest.TestCase):
    
    def testconvert_to_numeric(self):
        self.assertEqual(stock_base.convert_to_numeric("100M"), 100000000)
        self.assertEqual(stock_base.convert_to_numeric("100B"), 100000000000)
        self.assertEqual(stock_base.convert_to_numeric("100"), 100.0)
        self.assertEqual(stock_base.convert_to_numeric("100.5"), 100.5)
        self.assertEqual(stock_base.convert_to_numeric("100,000"), 100000.0)
        self.assertEqual(stock_base.convert_to_numeric("100.5%"), 0.1005)
        self.assertEqual(stock_base.convert_to_numeric(None), None)

class TestStockClass(test_utils.TestCaseBase):
    """Test stock_base class."""
    def setUp(self) -> None:
        """Create StockBase object and download all data.
        """
        super().setUp()
        self.stock.generate_more_data()
    
    def test_generate_more_data(self):
        # test calc_moving_average func
        for item in ['SMA5', 'SMA10', 'SMA20', 'SMA60', 'SMA120', 'SMA240']:
            self.assertTrue(item in self.stock.df)
        for item in ['EMA5', 'EMA10', 'EMA20', 'EMA60', 'EMA120', 'EMA240']:
            self.assertTrue(item in self.stock.df)
        # test calc_macd
        for item in ['MACD_DIF', 'MACD_DEM', 'MACD_OSC']:
            self.assertTrue(item in self.stock.df)


    def test_is_good_business(self) -> None:
        result = self.stock.is_good_business()
        self.assertFalse(result[0])

    def test_plot_valuation(self):
        self.stock.plot_valuation()
    
    def test_test(self):
        pass
    
if __name__ == "__main__":
    absltest.main()
