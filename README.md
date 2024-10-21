# Stabilized Proximal Point Methods for Federated Optimization [Neurips 2024 spotlight]

The official Pytorch implementation of [Stabilized Proximal Point Methods for Federated Optimization](https://openreview.net/forum?id=WukSyFSzDt&referrer=%5Bthe%20profile%20of%20Xiaowen%20Jiang%5D(%2Fprofile%3Fid%3D~Xiaowen_Jiang1)).

## Install dependencies
`pip install -r requirements.txt`

(The folder 'haven' is from [haven-ai](https://github.com/haven-ai/haven-ai).)

The implementation of the propsoed methods can be found in `./src/optimizers/sdane.py`.

## Experiments
`mkdir -p ./datasets/libsvm ./datasets/cifar10 ./figures ./results`

Follow or modify experiments in `exp_configs.py`. 

Modify `train.sh` and run: `bash train.sh`.

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


