from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime as dt
import logging
import os
from typing import KeysView

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_ta as ta
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import linregress

import algotrading
from algotrading import stock_base

logger = logging.getLogger(__name__)


class Stock(stock_base.StockBase):
    def __init__(self, name, description=None, **kwargs):
        super().__init__(name, description, **kwargs)

    def plot(self, **kwargs):
        super().plot(**kwargs)