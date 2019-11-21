# Unconstrained Monotonic Neural Networks (UMNN)
![](figures/archi.png)
Official implementation of Unconstrained Monotonic Neural Networks (UMNN) and the experiments presented in the paper:
> Antoine Wehenkel and Gilles Louppe. "Unconstrained Monotonic Neural Networks." (2019).
> [[arxiv]](https://arxiv.org/abs/1908.05164)

# Dependencies
The code has been tested with Pytorch 1.1 and Python3.6.
Some code to draw figures and load dataset are taken from 
[FFJORD](https://github.com/rtqichen/ffjord) 
and [Sylvester normalizing flows for variational inference](https://github.com/riannevdberg/sylvester-flows).

# Usage
## Simple Monotonic Function
This experiment is not describes in the paper. We create the following dataset:
x = [x_1, x_2, x_3] is drawn from a multivariate Gaussian, y = x_1^3 + x_2 + sin(x_3). 
We suppose we are given the information about which variable y is monotonic 
with respect with, here x_1.
```bash
python MonotonicMLP.py 
```

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
You have to download the datasets with the following command:
```bash
python datasets/download_datasets.py 
```
Then you can execute:
```bash
python UCIExperiments.py --data ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']
```
See UCIExperiments.py for optional arguments.

## VAE
You have to download the datasets:
* MNIST: 
```
python datasets/download_datasets.py
```
* OMNIGLOT: the dataset can be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset can be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).
* Frey Faces: the dataset can be downloaded from [link](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl).
```bash
python TrainVaeFlow.py -d ['mnist', 'freyfaces', 'omniglot', 'caltech']
```

## Other Usage
All the files related to the implementation of UMNN (Conditionner network, Integrand Network and Integral)
are located in the folder models/UMNN. 
- `NeuralIntegral.py` computes the integral of a neural network
(with 1d output) using the Clenshaw-Curtis(CC) quadrature, it computes sequentially the different evaluation points required by CC.
- `ParallelNeuralIntegral.py` processes all the evaluation points at once making the computation almost as fast as the forward evaluation 
the net but to the price of a higher memory cost. 
- `UMNNMAF.py` contains the implementation of the different networks required by UMNN.
- `UMNNMAFFlow.py` contains the implementation of flows made of UMNNs.