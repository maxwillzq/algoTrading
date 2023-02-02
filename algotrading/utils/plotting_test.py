from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from absl.testing import absltest

import algotrading
from algotrading.utils import plotting
from algotrading.utils import test_utils


class TestPlotting(test_utils.TestCaseBase):

  def test_plot_price_volume(self):
    df = self.stock.df
    plotting.plot_price_volume(df)
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
  absltest.main()
