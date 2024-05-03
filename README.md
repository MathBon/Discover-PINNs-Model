# Discovering-PINNs-Model
This repository contains the code for the paper: Discovering Physics-Informed Neural Networks Model for Solving Partial Differential Equations through Evolutionary Computation, Swarm and Evolutionary Computation, 2024.


# Description 

The following five methods in the paper are implemented based on pytorch (version>=1.9).

(1) evolution with DPSTE strategy (evo-w/-DPSTE)

(2) evolution without DPSTE strategy (evo-w/o-DPSTE)

(3) random search

(4) Bayesian optimization based on FcNet (BO-FcNet) 

(5) Bayesian optimization based on FcResNet (BO-FcResNet)

Three PDEs including Klein-Gordon equation, Burgers equation, and Lamé equations are introduced for experiment. 



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

