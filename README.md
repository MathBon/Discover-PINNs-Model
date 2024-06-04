# Discover-PINNs-Model
This repository contains the code for the paper: Bo Zhang, Chao Yang. Discovering Physics-Informed Neural Networks Model for Solving Partial Differential Equations through Evolutionary Computation, Swarm and Evolutionary Computation, 88, 2024, 101589.


# Description 

The following five methods in the paper are implemented based on pytorch (version>=1.9).

(1) evolution with DPSTE strategy (evo-w/-DPSTE)

(2) evolution without DPSTE strategy (evo-w/o-DPSTE)

(3) random search

(4) Bayesian optimization based on FcNet (BO-FcNet) 

(5) Bayesian optimization based on FcResNet (BO-FcResNet)

Three PDEs including Klein-Gordon equation, Burgers equation, and Lamé equations are solved in experiment. 



# Requirements

`pip install numpy`

`pip install bayesian-optimization`

`pip install matplotlib`



# Usage

Take discovering PINNs model for solving Klein-Gordon equation as an example: 

## Evolution and random search 

Evo-w/-DPSTE, evo-w/o-DPSTE, and random search are executed from **klein_gordon_evo_main.py**. 

The 'pattern' can be chosen as 'evolution' or 'random_search' in **klein_gordon_evo_main.py**.

The DPSTE strategy and hyperparameters in evolution are set in **config.py**.

## Bayesian optimization

BO-FcNet and BO-FcResNet are executed from **klein_gordon_bayes.py**.

## Evaluation

The discovered models are evaluated by **klein_gordon_pinns.py**.



# Citation
``bibtex
@article{zhang2024discovering,
  title={Discovering physics-informed neural networks model for solving partial differential equations through evolutionary computation},
  author={Zhang, Bo and Yang, Chao},
  journal={Swarm and Evolutionary Computation},
  volume={88},
  pages={101589},
  year={2024},
  publisher={Elsevier}
}

