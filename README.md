﻿

## Tested Environment

- Ubuntu 22.04
- [Python 3.7.16](https://www.anaconda.com/products/individual#Downloads)
- [Sklearn 1.0.2](https://scikit-learn.org/stable/install.html)
- [Pytorch 1.12.1](https://pytorch.org/get-started/locally/#linux-installation)
- [Numpy 1.21.6](https://numpy.org/install/)
- [Torch_geometric 2.3.1](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [Scipy 1.7.3](https://scipy.org/)
- [Dgl 0.9.0](https://www.dgl.ai/pages/start.html)

please create an env:
```
conda creare -n space_aml python=3.7.16
conda activate space_aml
```

install libraries:
```
pip install -r requirements.txt
```

## Datasets

Download data files from [GADbench](https://github.com/squareRoot3/GADBench) (or import from dgl) and put them in datasets/. 

**Directory Structure**

```
├── datasets
│   ├── amazon
│   │   ├── amazon (different datasets may have different types)
│   ├── main.py  
│   ├── utils.py
│   ├── name.py
```
Use main.py to generate training/validation/test set. 

**Example**
```
python main.py --data amazon
```

## Experiments

**Parameters**
As described in paper. 

**Example**
```
python main.py --data amazon
```
