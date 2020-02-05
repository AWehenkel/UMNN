"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
Modified by Antoine Wehenkel
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ------------------------------------------------------------------------------


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False, random=False, device="cpu"):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        
        super().__init__()
        self.random = random
        self.nin = nin
        self.nout = nout
        self.device = device
        self.pi = torch.tensor(math.pi).to(self.device)
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        
        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net).to(device)
        
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample the order of the inputs and the connectivity of all neurons
        if self.random:
            self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
            for l in range(L):
                self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])
        else:
            self.m[-1] = np.arange(self.nin)
            for l in range(L):
                self.m[l] = np.array([self.nin - 1 - (i % self.nin) for i in range(self.hidden_sizes[l])])
        
        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
        
        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)

        # map between in_d and order
        self.i_map = self.m[-1].copy()
        for k in range(len(self.m[-1])):
            self.i_map[self.m[-1][k]] = k

    
    def forward(self, x, context=None):
        if self.nout == 2:
            transf = self.net(x)
            mu, sigma = transf[:, :self.nin], transf[:, self.nin:]
            z = (x - mu) * torch.exp(-sigma)
            return z
        return self.net(x)

    def compute_ll(self, x):
        # Jac and x of MADE
        transf = self.net(x)
        mu, sigma = transf[:, :self.nin], transf[:, self.nin:]
        z = (x - mu) * torch.exp(-sigma)

        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = - sigma.sum(1) + log_prob_gauss

        return ll, z

    def invert(self, z):
        if self.nin != self.nout/2:
            return None

        # We suppose a Gaussian MADE
        u = torch.zeros(z.shape)
        for d in range(self.nin):
            transf = self.forward(u)
            mu, sigma = transf[:, self.i_map[d]], transf[:, self.nin + self.i_map[d]]
            u[:, self.i_map[d]] = z[:, self.i_map[d]] * torch.exp(sigma) + mu
        return u
# ------------------------------------------------------------------------------


class ConditionnalMADE(MADE):

    def __init__(self, nin, cond_in, hidden_sizes, nout, num_masks=1, natural_ordering=False, random=False, device="cpu"):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__(nin + cond_in, hidden_sizes, nout, num_masks, natural_ordering, random, device)
        self.nin_non_cond = nin
        self.cond_in = cond_in

    def forward(self, x, context):
        out = super().forward(torch.cat((context, x), 1))
        out = out.contiguous().view(x.shape[0], int(out.shape[1]/self.nin), self.nin)[:, :, self.cond_in:].contiguous().view(x.shape[0], -1)
        return out

    def computeLL(self, x, context):
        # Jac and x of MADE
        transf = self.net(torch.cat((context, x), 1))
        transf = transf.contiguous().view(x.shape[0], int(transf.shape[1] / self.nin), self.nin)[:, :, self.cond_in:].contiguous().view(x.shape[0], -1)
        mu, sigma = transf[:, :self.nin], transf[:, self.nin:]
        z = (x - mu) * torch.exp(-sigma)

        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = - sigma.sum(1) + log_prob_gauss

        return ll, z

    def invert(self, z, context):
        if self.nin != self.nout / 2:
            return None

        # We suppose a Gaussian MADE
        u = torch.zeros(z.shape)
        for d in range(self.nin):
            transf = self.net(torch.cat((context, x), 1))
            mu, sigma = transf[:, self.i_map[d]], transf[:, self.nin + self.i_map[d]]
            u[:, self.i_map[d]] = z[:, self.i_map[d]] * torch.exp(sigma) + mu
        return u


if __name__ == '__main__':
    from torch.autograd import Variable
    
    # run a quick and dirty test for the autoregressive property
    D = 10
    rng = np.random.RandomState(14)
    x = (rng.rand(1, D) > 0.5).astype(np.float32)
    
    configs = [
        (D, [], D, False),                 # test various hidden sizes
        (D, [200], D, False),
        (D, [200, 220], D, False),
        (D, [200, 220, 230], D, False),
        (D, [200, 220], D, True),          # natural ordering test
        (D, [200, 220], 2*D, True),       # test nout > nin
        (D, [200, 220], 3*D, False),       # test nout > nin
    ]
    
    for nin, hiddens, nout, natural_ordering in configs:
        
        print("checking nin %d, hiddens %s, nout %d, natural %s" % 
             (nin, hiddens, nout, natural_ordering))
        model = MADE(nin, hiddens, nout, natural_ordering=natural_ordering)
        z = torch.randn(1, nin)
        model.invert(z)
        continue
        # run backpropagation for each dimension to compute what other
        # dimensions it depends on.
        res = []
        for k in range(nout):
            xtr = Variable(torch.from_numpy(x), requires_grad=True)
            xtrhat = model(xtr)
            loss = xtrhat[0,k]
            loss.backward()
            
            depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % nin not in depends_ix
            
            res.append((len(depends_ix), k, depends_ix, isok))
        
        # pretty print the dependencies
        res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))
    
