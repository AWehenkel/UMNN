import torch
import numpy as np
import math


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float()

    return cc_weights, steps


def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None, inv_f=False):
    #Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    device = x0.get_device() if x0.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)

    if compute_grad:
        g_param = 0.
        g_h = 0.
    else:
        z = 0.
    xT = x0 + nb_steps*step_sizes
    for i in range(nb_steps + 1):
        x = (x0 + (xT - x0)*(steps[i] + 1)/2)
        if compute_grad:
            dg_param, dg_h = computeIntegrand(x, h, integrand, x_tot*(xT - x0)/2, inv_f)
            g_param += cc_weights[i]*dg_param
            g_h += cc_weights[i]*dg_h
        else:
            if inv_f:
                dz = 1/integrand(x, h)
            else:
                dz = integrand(x, h)
            z = z + cc_weights[i]*dz

    if compute_grad:
        return g_param, g_h

    return z*(xT - x0)/2


def computeIntegrand(x, h, integrand, x_tot, inv_f=False):
    with torch.enable_grad():
        if inv_f:
            f = 1/integrand.forward(x, h)
        else:
            f = integrand.forward(x, h)
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True))
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    return g_param, g_h


class NeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20, inv_f=False):
        with torch.no_grad():
            x_tot = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, h, False, inv_f=inv_f)
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone(), h)
            ctx.inv_f = inv_f
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, h = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        inv_f = ctx.inv_f
        integrand_grad, h_grad = integrate(x0, nb_steps, x/nb_steps, integrand, h, True, grad_output, inv_f)
        x_grad = integrand(x, h)
        x0_grad = integrand(x0, h)
        # Leibniz formula
        return -x0_grad*grad_output, x_grad*grad_output, None, integrand_grad, h_grad.view(h.shape), None
