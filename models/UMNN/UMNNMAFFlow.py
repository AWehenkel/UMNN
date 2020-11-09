import torch
import torch.nn as nn
from .UMNNMAF import EmbeddingNetwork, UMNNMAF
import numpy as np
import math


class ListModule(object):
    def __init__(self, module, prefix, *args):
        """
        The ListModule class is a container for multiple nn.Module.
        :nn.Module module: A module to add in the list
        :string prefix:
        :list of nn.module args: Other modules to add in the list
        """
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class UMNNMAFFlow(nn.Module):

    def __init__(self, nb_flow=1, nb_in=1, hidden_derivative=[50, 50, 50, 50], hidden_embedding=[50, 50, 50, 50],
                 embedding_s=20, nb_steps=50, act_func='ELU', solver="CC", cond_in=0, device="cpu"):
        """
        UMNNMAFFlow class is a normalizing flow made of UMNNMAF blocks.
        :int nb_flow: The number of components in the flow
        :int nb_in: The size of the input dimension (data)
        :list(int) hidden_derivative: The size of hidden layers in the integrand networks
        :list(int) hidden_embedding: The size of hidden layers in the embedding networks
        :int embedding_s: The size of the embedding
        :int nb_steps: The number of integration steps (0 for random)
        :string solver: The solver (CC or CCParallel)
        :int cond_in: The size of the conditionning variable
        :string device: The device (cpu or gpu)
        """
        super().__init__()
        self.device = device
        self.register_buffer("pi", torch.tensor(math.pi))
        self.nets = ListModule(self, "Flow")
        for i in range(nb_flow):
            auto_net = EmbeddingNetwork(nb_in, hidden_embedding, hidden_derivative, embedding_s, act_func=act_func,
                                                     device=device, cond_in=cond_in).to(device)

            model = UMNNMAF(auto_net, nb_in, nb_steps, device, solver=solver).to(device)
            self.nets.append(model)

    def to(self, device):
        for net in self.nets:
            net.to(device)
        self.device = device
        super().to(device)
        return self

    def forward(self, x, context=None):
        inv_idx = torch.arange(x.size(1) - 1, -1, -1).long()
        for net in self.nets:
            x = net.forward(x, context=context)[:, inv_idx]
        return x[:, inv_idx]

    def invert(self, z, context=None):
        """
        From image to domain.
        :param z: A tensor of noise.
        :param iter: The number of iteration (accuracy should be around 25/100**iter
        :param context: Conditioning variable
        :return: Domain value
        """
        inv_idx = torch.arange(z.size(1) - 1, -1, -1).long()
        z = z[:, inv_idx]
        for net_i in range(len(self.nets)-1, -1, -1):
            z = self.nets[net_i].invert(z[:, inv_idx], context=context)
        return z

    def compute_log_jac(self, x, context=None):
        log_jac = 0.
        inv_idx = torch.arange(x.size(1) - 1, -1, -1).long()
        for net in self.nets:
            log_jac += net.compute_log_jac(x, context=context)
            x = net.forward(x, context=context)[:, inv_idx]
        return log_jac

    def compute_log_jac_bis(self, x, context=None):
        log_jac = 0.
        inv_idx = torch.arange(x.size(1) - 1, -1, -1).long()
        for net in self.nets:
            x, l = net.compute_log_jac_bis(x, context=context)
            x = x[:, inv_idx]
            log_jac += l
        return x[:, inv_idx], log_jac

    def compute_ll(self, x, context=None):
        log_jac = 0.
        inv_idx = torch.arange(x.size(1) - 1, -1, -1).long()
        for net in self.nets:
            z = net.forward(x, context=context)[:, inv_idx]
            log_jac += net.compute_log_jac(x, context=context)
            x = z
        z = z[:, inv_idx]
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = log_jac.sum(1) + log_prob_gauss
        return ll, z

    def compute_ll_bis(self, x, context=None):
        log_jac = 0.
        inv_idx = torch.arange(x.size(1) - 1, -1, -1).long()
        for net in self.nets:
            log_jac += net.compute_log_jac(x, context=context)
            x = net.forward(x, context=context)[:, inv_idx]
        z = x[:, inv_idx]
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2)
        ll = log_jac + log_prob_gauss
        return ll, z

    def compute_bpp(self, x, alpha=1e-6, context=None):
        d = x.shape[1]
        ll, z = self.compute_ll(x, context=context)
        bpp = -ll / (d * np.log(2)) - np.log2(1 - 2 * alpha) + 8 \
              + 1 / d * (torch.log2(torch.sigmoid(x)) + torch.log2(1 - torch.sigmoid(x))).sum(1)
        return bpp, ll, z

    def set_steps_nb(self, nb_steps):
        for net in self.nets:
            net.set_steps_nb(nb_steps)

    def compute_lipschitz(self, nb_iter=10):
        L = 1.
        for net in self.nets:
            L *= net.compute_lipschitz(nb_iter)
        return L

    def force_lipschitz(self, L=1.5):
        for net in self.nets:
            net.force_lipschitz(L)
