# Efficient gradient-based methods for bilevel learning via recycling Krylov subspaces
Solve a sequence of linear systems employing RMINRES, a recycling Krylov subspace method. Compare recycling strategies based on (harmonic) Ritz vectors and a new quantity, Ritz generalized singular vectors.
The linear systems arise from solving a bilevel learning problem of determining optimal convolution filters of the fields of experts regularizer.

## Installation
The main dependencies are:
* Python [3.9.20]
* [Numpy](https://pypi.org/project/numpy/) [1.26.4]
* [PyTorch](https://pytorch.org/) [2.5.1]
* [Scipy](https://pypi.org/project/scipy/) [1.13.1]
* [Matplotlib](https://pypi.org/project/matplotlib/) [3.9.2]
* [TQDM](https://pypi.org/project/tqdm/) [4.66.5]
* [Pandas](https://pypi.org/project/pandas/) [2.2.3]

and a full virtual environment can be created using the [environment.yaml](environment.yaml) file in conda by running the line
```
conda env create -f environment.yaml
```
in the terminal.

## Getting started
To recreate the results and plots associated with the BSDS300 dataset experiment:
 * [BSDS_generate_bilevel_data.py](BSDS_generate_bilevel_data.py) solves the bilevel learning problem
 * [BSDS_generate_recycle_data.py](BSDS_generate_recycle_data.py) re-solves the sequence of Hessian systems encountered in the bilevel learning problem, employing numerous recycling Krylov subspace strategies. 
 * [BSDS_plot.py](BSDS_plot.py) displays all results associated with the different recycling strategies.
 
To recreate the plots associated with the MNIST datset experiments:
* [MNIST_plot.py](MNIST_plot.pt) displays all results associated with the different recycling strategies



