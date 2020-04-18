import torch
import torch.nn as nn
from .NeuralIntegral import NeuralIntegral
from .ParallelNeuralIntegral import ParallelNeuralIntegral
import numpy as np
import math
from .made import MADE, ConditionnalMADE
class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()
    def forward(self, x):
        return self.elu(x) + 1.


dict_act_func = {"Sigmoid": nn.Sigmoid(), "ELU": ELUPlus()}

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_lipschitz_linear(W, nb_iter=10):
    x = torch.randn(W.shape[1], 1).to(W.device)
    for i in range(nb_iter):
        x_prev = x
        x = W.transpose(0, 1) @ (W @ x_prev)
        x = x/torch.norm(x)

    lam = (torch.norm(W.transpose(0, 1) @ (W @ x))/torch.norm(x))**.5
    return lam


class UMNNMAF(nn.Module):
    def __init__(self, net, input_size, nb_steps=100, device="cpu", solver="CC"):
        super().__init__()
        self.net = net.to(device)
        self.device = device
        self.input_size = input_size
        self.nb_steps = nb_steps
        self.cc_weights = None
        self.steps = None
        self.solver = solver
        self.register_buffer("pi", torch.tensor(math.pi))

        # Scaling could be changed to be an autoregressive network output
        self.scaling = nn.Parameter(torch.zeros(input_size, device=self.device), requires_grad=False)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, x, method=None, x0=None, context=None):
        x0 = x0.to(x.device) if x0 is not None else torch.zeros(x.shape).to(x.device)
        xT = x
        h = self.net.make_embeding(xT, context)
        z0 = h.view(h.shape[0], -1, x.shape[1])[:, 0, :]

        # s is a scaling factor.
        s = torch.exp(self.scaling.unsqueeze(0).expand(x.shape[0], -1))
        if self.solver == "CC":
            z = NeuralIntegral.apply(x0, x, self.net.parallel_nets, _flatten(self.net.parallel_nets.parameters()),
                                     h, self.nb_steps) + z0
        elif self.solver == "CCParallel":
            z = ParallelNeuralIntegral.apply(x0, x, self.net.parallel_nets, _flatten(self.net.parallel_nets.parameters()),
                                     h, self.nb_steps) + z0
        else:
            return None
        return s*z

    def compute_cc_weights(self, nb_steps):
        lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
        lam = np.cos((lam @ lam.T)*math.pi/nb_steps)
        lam[:, 0] = .5
        lam[:, -1] = .5*lam[:, -1]
        lam = lam*2/nb_steps
        W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
        W[np.arange(1, nb_steps + 1, 2)] = 0
        W = 2/(1 - W**2)
        W[0] = 1
        W[np.arange(1, nb_steps + 1, 2)] = 0
        self.cc_weights = torch.tensor(lam.T @ W).float().to(self.device)
        self.steps = torch.tensor(np.cos(np.arange(0, nb_steps+1, 1).reshape(-1, 1) * math.pi/nb_steps)).float().to(self.device)

    def compute_log_jac(self, x, context=None):
        self.net.make_embeding(x, context)
        jac = self.net.forward(x)
        return torch.log(jac + 1e-10) + self.scaling.unsqueeze(0).expand(x.shape[0], -1)

    def compute_log_jac_bis(self, x, context=None):
        z = self.forward(x, context=context)
        jac = self.net.forward(x)
        return z, torch.log(jac + 1e-10) + self.scaling.unsqueeze(0).expand(x.shape[0], -1)

    def compute_ll(self, x, context=None):
        z = self.forward(x, context=context)
        jac = self.net.forward(x)

        z.clamp_(-10., 10.)
        log_prob_gauss = -.5 * (torch.log(self.pi * 2) + z ** 2).sum(1)
        ll = log_prob_gauss + torch.log(jac + 1e-10).sum(1) + self.scaling.unsqueeze(0).expand(x.shape[0], -1).sum(1)

        return ll, z

    def compute_ll_bis(self, x, context=None):
        z = self.forward(x, context=context)
        jac = self.net.forward(x)

        ll = torch.log(jac + 1e-10) + self.scaling.unsqueeze(0).expand(x.shape[0], -1)
        z.clamp_(-10., 10.)
        return ll, z

    def compute_bpp(self, x, alpha=1e-6, context=None):
        d = x.shape[1]
        ll, z = self.computeLL(x, context=context)
        bpp = -ll/(d*np.log(2)) - np.log2(1 - 2*alpha) + 8 \
              + 1/d * (torch.log2(torch.sigmoid(x)) + torch.log2(1 - torch.sigmoid(x))).sum(1)
        z.clamp_(-10., 10.)
        return bpp, ll, z

    def set_steps_nb(self, nb_steps):
        self.nb_steps = nb_steps

    def compute_lipschitz(self, nb_iter=10):
        return self.net.parallel_nets.computeLipshitz(nb_iter)

    def force_lipschitz(self, L=1.5):
        self.net.parallel_nets.force_lipschitz(L)

    # Kind of dichotomy with a factor 100.
    def invert(self, z, iter=10, context=None):
        nb_step = 10
        step = 1/(nb_step - 1)

        x_range = (torch.ones(z.shape[0], nb_step) * torch.arange(0, 1 + step/2, step)).permute(1, 0).to(self.device)
        z = z.unsqueeze(0).expand(nb_step, -1, -1)
        x = z.clone()
        x_inv = torch.zeros(z.shape[1], z.shape[2]).to(self.device)
        left, right = -50*torch.ones(z.shape[1], z.shape[2]).to(self.device), torch.ones(z.shape[1], z.shape[2])\
            .to(self.device)*50
        s = torch.exp(self.scaling.unsqueeze(0).unsqueeze(1).expand(x.shape[0], x.shape[1], -1))
        with torch.no_grad():
            for j in range(self.input_size):
                if j % 100 == 0:
                    print(j)

                # Compute embedding and keep only the one related to x_j
                h = self.net.make_embeding(x_inv, context)
                offset = h.view(x_inv.shape[0], -1, x_inv.shape[1])[:, 0, [j]]
                h_idx = torch.arange(j, h.shape[1], z.shape[2]).to(self.device)
                h = h[:, h_idx]

                h, offset = h.squeeze(1).unsqueeze(0).expand(nb_step, -1, -1), offset.unsqueeze(0).expand(nb_step, -1, -1)
                x0 = torch.zeros(offset.shape).view(-1, 1).to(self.device)

                derivative = lambda x, h: self.net.parallel_nets.independant_forward(torch.cat((x, h), 1))

                for i in range(iter):
                    x[:, :, j] = x_range * (right[:, j] - left[:, j]) + left[:, j]
                    # if i == 0:
                    #     print(right[:, j], left[:, j])
                    z_est = s[:, :, [j]]*(offset + ParallelNeuralIntegral.apply(x0, x[:, :, j].contiguous().view(-1, 1),
                                                                                derivative, None,
                                                                                h.contiguous().view(x0.shape[0], -1),
                                                                                self.nb_steps).contiguous().view(nb_step, -1, 1))

                    _, z_pos = torch.abs(z_est[:, :, 0] - z[:, :, j]).min(0)

                    pos_midle = z_pos + torch.arange(0, z.shape[1]).to(self.device)*nb_step
                    z_val = z_est[:, :, 0].t().contiguous().view(-1)[pos_midle]
                    x_flat = x[:, :, j].t().contiguous().view(-1)

                    mask = (z_val < z[0, :, j]).float()

                    pos_left = pos_midle - 1
                    pos_right = (pos_midle + 1) % x_flat.shape[0]

                    left[:, j] = (mask * x_flat[pos_midle] + (1 - mask) * x_flat[pos_left])
                    right[:, j] = (mask * x_flat[pos_right] + (1 - mask) * x_flat[pos_midle])
                x_inv[:, j] = x_flat[pos_midle]
        return x_inv


class IntegrandNetwork(nn.Module):
    def __init__(self, nnets, nin, hidden_sizes, nout, act_func='ELU', device="cpu"):
        super().__init__()
        self.nin = nin
        self.nnets = nnets
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.device = device

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.LeakyReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(dict_act_func[act_func])
        self.net = nn.Sequential(*self.net)
        self.masks = torch.eye(nnets).to(device)

    def to(self, device):
        self.device = device
        self.net.to(device)
        self.masks.to(device)
        return self

    def forward(self, x, h):
        x = torch.cat((x, h), 1)
        nb_batch, size_x = x.shape
        x_he = x.view(nb_batch, -1, self.nnets).transpose(1, 2).contiguous().view(nb_batch*self.nnets, -1)
        y = self.net(x_he).view(nb_batch, -1)
        return y

    def independant_forward(self, x):
        return self.net(x)

    def compute_lipschitz(self, nb_iter=10):
        with torch.no_grad():
            L = 1
            for layer in self.net.modules():
                if isinstance(layer, nn.Linear):
                    L *= compute_lipschitz_linear(layer.weight, nb_iter)
        return L

    def force_lipschitz(self, L=1.5):
        with torch.no_grad():
            for layer in self.net.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight /= max(compute_lipschitz_linear(layer.weight, 10)/L, 1)


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_d, hiddens_embedding=[50, 50, 50, 50], hiddens_integrand=[50, 50, 50, 50], out_made=1,
                 cond_in=0, act_func='ELU', device="cpu"):
        super().__init__()
        self.m_embeding = None
        self.device = device
        self.in_d = in_d
        if cond_in > 0:
            self.made = ConditionnalMADE(in_d, cond_in, hiddens_embedding, (in_d + cond_in) * (out_made), num_masks=1,
                                         natural_ordering=True).to(device)
        else:
            self.made = MADE(in_d, hiddens_embedding, in_d * (out_made), num_masks=1, natural_ordering=True).to(device)
        self.parallel_nets = IntegrandNetwork(in_d, 1 + out_made, hiddens_integrand, 1, act_func=act_func, device=device)

    def to(self, device):
        self.device = device
        self.made.to(device)
        self.parallel_nets.to(device)
        return self

    def make_embeding(self, x_made, context=None):
        self.m_embeding = self.made.forward(x_made, context)
        return self.m_embeding

    def forward(self, x_t):
        return self.parallel_nets.forward(x_t, self.m_embeding)

