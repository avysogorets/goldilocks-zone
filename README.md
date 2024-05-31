## [Deconstructing the Goldilocks Zone of Neural Network Initialization](https://arxiv.org/abs/2402.03579)
This repo contains code to reproduce experiments from the paper.
## Quick setup
Follow these steps in your terminal window to set up virtual environment:
#### MacOS & Linux
```
python -m pip install --user --upgrade pip # install pip
python -m pip install --user virtualenv # install environment manager
python -m venv env # create a new environment
source env/bin/activate # activate the environment
python -m pip install -r requirements.txt # install packages
```
## Usage
The supported models include LeNet-300-100 (FashionMNIST) and LeNet-5 (CIFAR-10). For demonstration
purposes, we recommend using model ```Demo``` and a small 2D dataset ```Circles```. See the Jupyter notebook for
most of the experiments. To replicate the empirical work from Section 5, please see ```trainability.py``` and the
associated command line arguments.

## Cite us
```
@InProceedings{vysogorets2024deconstructing,
title = {Deconstructing the Goldilocks Zone of Neural Network Initialization},
author = {Vysogorets, Artem and Dawid, Anna and Kempe, Julia},
booktitle = {Proceedings of the 41st International Conference on Machine Learning},
pages = {1--15},
year = {2024},
series = {Proceedings of Machine Learning Research},
month = {21--27 Jul},
publisher = {PMLR}}
```
