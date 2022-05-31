---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## stock_info utility functions

```python
import algotrading
from algotrading import stock_info as si
```

```python
print(dir(si))
```

```python
len(si.tickers_sp500())
len(si.tickers_dow())
len(si.tickers_other())
len(si.tickers_nasdaq())
```

```python
#si.get_analysts_info('AMZN')
#si.get_balance_sheet('AMZN')
#si.get_cash_flow('AMZN')
#si.get_company_info('AMZN')
#si.get_company_officers('AMZN')
#si.get_currencies()
#si.get_data('AMZN')
#si.get_day_gainers()
#si.get_day_losers()
#si.get_day_most_active()
#si.get_dividends('TSM')
#si.get_earnings('AMZN')
#si.get_earnings_for_date("06/01/2022")
#si.get_earnings_history('AMZN')
#si.get_financials('AMZN')
#si.get_futures()
#si.get_holders('AMZN')
#si.get_income_statement('AMZN')
#si.get_live_price('AMZN')
#si.get_market_status()
#si.get_postmarket_price('AMZN')
#si.get_premarket_price('AMZN')
#si.get_quote_data('AMZN')
#si.get_quote_table('AMZN')
#si.get_splits('AMZN')
#si.get_stats('AMZN')
#si.get_stats_valuation('AMZN')
#si.get_top_crypto()
#si.get_undervalued_large_caps()
```
