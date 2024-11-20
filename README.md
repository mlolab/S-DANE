# Stabilized Proximal Point Methods for Federated Optimization [Neurips 2024 spotlight]

The official Pytorch implementation of [Stabilized Proximal Point Methods for Federated Optimization](https://openreview.net/forum?id=WukSyFSzDt&referrer=%5Bthe%20profile%20of%20Xiaowen%20Jiang%5D(%2Fprofile%3Fid%3D~Xiaowen_Jiang1)).

## 1. Requirement
1)  Install the package [Poetry](https://python-poetry.org) following the [official poetry installation instructions](https://python-poetry.org/docs/).

2)  Navigate to the code directory 

3) Install dependencies defined in `pyproject.toml`
   ```
   poetry install
   ```

4) Test installation on a simple polyhedron feasibility problem
   ```
   poetry run pytest -s
   ```

## 2. Usage
### 2.1 Optimizers

The implementation of our propsoed methods (both standard and accelerated) can be found in `./src/optimizers/sdane.py`.

Besides, we also provide implementations of 1) [Acc-Extragradient](https://arxiv.org/pdf/2205.15136), 2) [DANE](https://arxiv.org/pdf/1312.7853), 3) [Scaffold](https://arxiv.org/pdf/1910.06378), 4) [Scaffnew](https://arxiv.org/pdf/2202.09357), 5) [FedRed](https://proceedings.mlr.press/v235/jiang24e.html), 6) [FedProx](https://arxiv.org/pdf/1812.06127), and 7) [FedAvg](https://arxiv.org/abs/1602.05629). These can be found in 
`./src/optimizers/fedred.py`

### 2.2 Experiments
The configurations of some pre-defined experiments can be found in `exp_configs.py`.

Here is an example of how to run the experiment of logistic regression on the dataset of ijcnn.
```
mkdir -p ./datasets/libsvm ./figures ./results

poetry run python trainval.py -e=ijcnn_iid_adaptive -sb=./results -d=./datasets/libsvm -c=0
```

## Visualization

Use `plot.ipynb` for illustration of the results.


## Citation
```bibtex
@article{jiang2024stabilized, 
  title={Stabilized proximal-point methods for federated optimization}, 
  author={Jiang, Xiaowen and Rodomanov, Anton and Stich, Sebastian U}, 
  journal={arXiv preprint arXiv:2407.07084}, 
  year={2024}}
```


