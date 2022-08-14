from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime as dt
import logging
import os

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
from scipy.stats import linregress

import algotrading
from algotrading import stock_base

logger = logging.getLogger(__name__)

class Fred(stock_base.StockBase):
    def __init__(self, name, description=None):
        super().__init__(name, description)

    def read_data(self, *args, **kwargs):
        """Read Stock Data from Yahoo Finance
        """
        super().read_data(*args, **kwargs)

        round_df = self.df.round(4)
        self.markdown_notes += f"{round_df.tail(5).to_markdown()}\n"
        return self.df
    
    def generate_more_data(self, days=14):
        pass

    def plot(self, **kwargs):
        last = len(self.df) - 1
        delta = 1
        daily_percentage = (self.df[self.name].iloc[last] - self.df[self.name].iloc[last - delta])/self.df[self.name].iloc[last - delta] * 100
        daily_percentage = round(daily_percentage, 2)
        plt.figure(figsize=(12,9))
        df = self.df.copy()
        df.plot(y=[self.name])
        plt.legend(loc='upper left', shadow=True, fontsize='x-large')
        plt.grid()
        plt.title(f"Today's increase={daily_percentage}%")

        if 'result_dir' in kwargs:
            result_dir = kwargs['result_dir']
            file_name = os.path.join(result_dir, self.name + ".png")
            plt.savefig(file_name,dpi=300)
            plt.close()
            self.markdown_notes += f"![{self.name}]({self.name}.png)\n\n\n"
            return file_name
        else:
            fig = plt.gcf()
            return fig
    
    def get_price_change_table(self):
        """generate result_dict dict.
        key is "5D%, 10D% ..." or moving average "20MA% .."
        value is difference to this baseline
        """
        result_dict = {}
        result_dict["name"] = self.name
        if self.description:
            result_dict["description"] = self.description
        last = len(self.df) - 1
        for delta in [1, 5, 20, 60, 120, 240]:
            key_name = f"{delta}D%"
            if last - delta > 0:
                value = (self.df[self.name].iloc[last] - self.df[self.name].iloc[last - delta])/self.df[self.name].iloc[last - delta] * 100
                value = round(value, 2)
                result_dict[key_name] = value
            else:
                result_dict[key_name] = None
        
        return result_dict

    def to_markdown(self):
        return self.markdown_notes
