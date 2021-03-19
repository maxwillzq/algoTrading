from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import yaml
import os
import pandas as pd
logger = logging.getLogger(__name__)

def get_data_dict(file_name):
    TOP_DIR = os.path.dirname(__file__)
    full_file_path = os.path.join(TOP_DIR, file_name)
    with open(full_file_path, 'r') as f:
        result_dict = yaml.load(f, Loader=yaml.FullLoader)
    return result_dict

def get_data_dir():
    TOP_DIR = os.path.dirname(__file__)
    return TOP_DIR

def get_SP500_list():
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df['Symbol'] = df['Symbol'].apply(lambda str: str.replace('.', '-'))
    stk_list = list(df.Symbol)
    return stk_list


def get_most_active_stocks(number=10):
        
    url = f'https://finance.yahoo.com/screener/predefined/most_actives?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_most_shorted_stocks(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/most_shorted_stocks?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_day_gainers(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/day_gainers?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_day_losers(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/day_losers?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_undervalued_large_caps(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/undervalued_large_caps?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_aggressive_small_caps(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/aggressive_small_caps?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_solid_large_growth_funds(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/solid_large_growth_funds?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_growth_technology_stocks(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/growth_technology_stocks?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list

def get_undervalued_growth_stocks(number=10):
    
    url = f'https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks?count={number}&offset=0'
    data = pd.read_html(url)[0]
    stk_list = list(data.Symbol)
    return stk_list