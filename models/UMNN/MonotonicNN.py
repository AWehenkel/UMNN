import torch
import torch.nn as nn
from NeuralIntegral import NeuralIntegral
from ParallelNeuralIntegral import ParallelNeuralIntegral


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers, n_out=1):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [n_out]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.


class MonotonicNN(nn.Module):
    '''
    in_d : The total number of inputs
    hidden_layers : a list a the number of neurons, to be used by a network that compresses the non-monotonic variables and by the integrand net.
    nb_steps : Number of integration steps
    n_out : the number of output (each output will be monotonic w.r.t one variable)
    '''
    def __init__(self, in_d, hidden_layers, nb_steps=200, n_out=1, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers, n_out)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2 * n_out]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps
        self.n_out = n_out

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h are just other conditionning variables.
    '''
    def forward(self, x, h):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, :self.n_out]
        scaling = torch.exp(out[:, self.n_out:])
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset

    '''
    The inverse procedure takes as input y which is the variable for which the inverse must be computed, h are just other conditionning variables.
    One output per n_out.
    y should be a scalar.
    '''
    def inverse(self, y, h):
        idx = (torch.arange(0, self.n_out**2, self.n_out + 1).view(1, -1) + torch.arange(0, (self.n_out**2)*y.shape[0], self.n_out**2).view(-1, 1)).view(-1)
        out = self.net(h)
        offset = out[:, :self.n_out]
        scaling = torch.exp(out[:, self.n_out:])
        y = (y.expand(-1, self.n_out) - offset)/scaling
        y = y.view(-1, 1)
        h = h.unsqueeze(1).expand(-1, self.n_out, -1).contiguous().view(y.shape[0], -1)
        y0 = torch.zeros(y.shape).to(self.device)
        return ParallelNeuralIntegral.apply(y0, y, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps, True).view(-1)[idx].view(-1, self.n_out)


net = MonotonicNN(3, [50, 50, 50], n_out=3)
x = torch.arange(-2, 2, .1).view(-1, 1)
h = torch.zeros(x.shape[0], 2) + 1.
y = net(x, h)
x_est = net.inverse(y[:, [2]], h)
print(x_est[0:20, :], x[0:20, :])
print(x.shape, y.shape)
import matplotlib.pyplot as plt
#plt.plot(x.numpy(), y.detach().numpy())
#plt.show()