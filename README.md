# Efficient gradient-based methods for bilevel learning via recycling Krylov subspaces
Solve a sequence of linear systems employing RMINRES, a recycling Krylov subspace method. Compare recycling strategies based on (harmonic) Ritz vectors and a new quantity, Ritz generalized singular vectors.
The linear systems arise from solving a bilevel learning problem of determining optimal convolution filters of the fields of experts regularizer for an MNIST inpainting problem. 

## Installation
The main dependencies are:
* Python [3.9.20]
* [Numpy](https://pypi.org/project/numpy/) [1.26.4]
* [PyTorch](https://pytorch.org/) [2.5.1]
* [Scipy](https://pypi.org/project/scipy/) [1.13.1]
* [Matplotlib](https://pypi.org/project/matplotlib/) [3.9.2]
* [TQDM](https://pypi.org/project/tqdm/) [4.66.5]
* [Pandas](https://pypi.org/project/pandas/) [2.2.3]

and a full virtual environment can be created using the [environment.yml](environment.yml) file in conda by running the line
```
conda env create -f environment.yml
```
in the terminal.

## Getting started
 All figures and numerical results are already available. To recreate them, there are 3 main files:
 * `generate_bilevel_problem_solution.py` solves the bilevel learning problem associated with determining optimal filters for fields of experts regularizer of an inpainting MNIST problem.
 * `generate_results.py` re-solves the sequence of Hessian systems encountered in bilevel learning problem solve, employing numerous recycling Krylov  subspace strategies. 
 * `generate_plots.py` displays all results associated with the different recycling strategies.