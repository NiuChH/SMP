# Permutation-equivariant and Proximity-aware Graph Neural Networks with Stochastic Message Passing
This repository is the official PyTorch implementation of "Permutation-equivariant and Proximity-aware Graph Neural Networks with Stochastic Message Passing".

[IEEE TKDE](https://ieeexplore.ieee.org/document/9721559/authors#authors)

[Arxiv](https://arxiv.org/abs/2009.02562)


## Dependencies

* PyTorch (tested on `1.12.0+cu113`), please refer to [PyTorch official site](https://pytorch.org/) for installation 

* PyTorch-geometric (tested on `2.0.4`), please refer to [PyTorch-geometric offical site](https://pytorch-geometric.readthedocs.io/en/2.0.4/notes/installation.html) for installation
* other dependencies are listed in `requirements.txt`, please install them with `pip install -r requirements.txt `

For datasets:

`PPI`  and `Email` datasets are included in `data` folder. Please unzip ppi.zip first if you need to use `PPI`. The other datasets will automatically download and unzip when needed (thanks to the libraries `networkx` and `obg` )

## Run

 `main.py` is the entrance for a whole training-validation-testing process. Run `python main.py -h` for a full parameter list and information.

Alternatively, to run several tasks sequentially with more complex configures, please refer to `run_all.py` (to run all the tasks on all the dataset we adopt except PPA), and `run_ppa.py` (to run tasks on PPA). You can also refer to these files for suitable defualt parameter values.

## Cite This

```text
@ARTICLE{9721559,
  author={Zhang, Ziwei and Niu, Chenhao and Cui, Peng and Pei, Jian and Zhang, Bo and Zhu, Wenwu},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Permutation-equivariant and Proximity-aware Graph Neural Networks with Stochastic Message Passing}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TKDE.2022.3154391}}
```
