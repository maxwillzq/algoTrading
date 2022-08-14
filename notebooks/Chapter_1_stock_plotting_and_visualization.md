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

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/maxwillzq/algoTrading/blob/main/Chapter_1_stock_plotting_and_visualization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="8uM_VEGRu2O9" -->
# Portfolio Optimization and Algorithmic Trading
<!-- #endregion -->

<!-- #region id="8kdijMXloeeL" -->
## install software
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="A5FZMPqeoglP" outputId="2b3fd09c-cb78-4796-ed07-dd85e111255f"
#!pip install git+https://github.com/maxwillzq/algoTrading.git
```

```python
import algotrading
import matplotlib.pyplot as plt
# set plotting figure parameters
plt.rcParams["figure.figsize"] = (15,9)
```

<!-- #region id="hKhY2AKSidJE" -->
# Select stock from stock table

Step 1: choose **stock**
<!-- #endregion -->

```python id="cNQyznM2CW20"
stock = algotrading.stock.Stock("AMD")
stock.read_data(days=365)
stock.generate_more_data()
```

```python
stock.
```

```python colab={"base_uri": "https://localhost:8080/", "height": 850} id="WboV_3_cCsWO" outputId="b1fd386c-3b3e-4bb3-f00f-d16eba1bc697"
stock.plot_density()
```

```python
stock.plot_valuation()
```

```python

```

## 阅读宏观数据

```python
# cpi
fred = algotrading.fred.Fred('CPIAUCSL')
fred.read_data(days=2 * 365)
cpi = fred.df / fred.df.shift(12) - 1.0
cpi = cpi * 100.0
```

```python
cpi.plot(grid=True)
```

```python id="H6HLPMZ6CzvL"
# 10年期国债利率
fred = algotrading.stock.Stock('^TNX')
fred.read_data()
fred.df['Close'].plot(grid=True)

```

```python id="1AILiZWi7Vpz"
# EFFR: Effective Federal Rate
fred = algotrading.stock.Stock('EFFR')
fred.read_data()
fred.df.plot(grid=True)

```

```python
#RESPPANWW: FED open market holding
fred = algotrading.stock.Stock('RESPPANWW')
fred.read_data()
fred.df.plot(grid=True)

```

```python
# UNRATE: Unemployment Rate
fred = algotrading.stock.Stock('UNRATE')
fred.read_data(days=4*365)
fred.df.plot(grid=True)

```
