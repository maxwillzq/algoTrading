#!/bin/bash
set -x
python3 draw_single_plot.py --stock_list shuping --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list keyao  --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list etf --with_chart No --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_most_active_stocks --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_most_shorted_stocks --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_day_gainers --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_day_losers --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_undervalued_large_caps --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_aggressive_small_caps --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_growth_technology_stocks --days 200 --sort_by mid_term,short_term,1D%
python3 draw_single_plot.py --stock_list get_undervalued_growth_stocks --days 200 --sort_by mid_term,short_term,1D%

