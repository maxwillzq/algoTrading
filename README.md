# algoTrading

install pandoc 
```
sudo apt get install pandoc
```
=======

## User guide


## Developer guide

Run command 

```
python setup.py develop
```

Then run the algoTrading at_run command line interface to see help message

```
at_run --help
```

You can use unittest to pick specific test to run.

For example:
```
python -m unittest algotrading.utils.plotting_test.test_plot_price_density
```

### Profiling

here we use cProfile for profiling and snaviz for visualization. see the [link](https://medium.com/@narenandu/profiling-and-visualization-tools-in-python-89a46f578989)

```bash
python3 -m cProfile -o test.cprofile  ./algotrading/scripts/draw_single_plot.py --config /Users/johnqiangzhang/Documents/open_source/algoTrading/algotrading/scripts/save_visualization/input_config.yaml
snakeviz test.cprofile 
```

cprofilev is also good. 
```bash
pip install cprofilev
```

then 
```bash
cprofilev -f test.profile 
```



### CLI

### jupyter notebook

Assume you have the python virtual enviroment whose name is develop_py.
```source ~/Documents/develop_py/bin/activate```


```bash
python -m ipykernel install --user --name=develop_py
jupyter notebook
```