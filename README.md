# Unconstrained Monotonic Neural Network (UMNN)
![](figures/archi.png)
Official implementation of Unconstrained Monotonic Neural Network (UMNN) and the experiments presented in the paper:
> Antoine Wehenkel and Gilles Louppe. "Unconstrained Monotonic Neural Networks." (2019).
> [[arxiv]](https://arxiv.org/abs/1908.05164)

# Dependencies
The code has been tested with Pytorch 1.1 and Python3.6.
Some code to draw figures and load dataset are taken from 
[FFJORD](https://github.com/rtqichen/ffjord) 
and [Sylvester normalizing flows for variational inference](https://github.com/riannevdberg/sylvester-flows).

# Usage

## Toy Experiments
![](figures/toy/all_flow.png)
```bash
python ToyExperiments.py 
```
See ToyExperiments.py for optional arguments.
## MNIST
![](figures/MNIST/MNIST_3_075.png)
```bash
python MNISTExperiment.py
```
See MNISTExperiment.py for optional arguments.

## UCI Dataset
```bash
python UCIExperiments
```
See UCIExperiments.py for optional arguments.

## VAE
```bash
python TrainVaeFlow.py
```

## Other Usage
All the files related to the implementation of UMNN (Conditionner network, Integrand Network and Integral)
are located in the folder models/UMNN. 
- `NeuralIntegral.py computes the integral of a neural network
(with 1d output) using the Clenshaw-Curtis(CC) quadrature, it computes sequentially the different evaluation points required by CC.
- `ParallelNeuralIntegral.py` processes all the evaluation points at once making the computation almost as fast as the forward evaluation 
the net but to the price of a higher memory cost. 
- `UMNNMAF.py` contains the implementation of the different networks required by UMNN.
- `UMNNMAFFlow.py` contains the implementation of flows made of UMNNs.