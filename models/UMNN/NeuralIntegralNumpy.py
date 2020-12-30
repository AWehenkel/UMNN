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
    cc_weights = lam.T @ W
    steps = np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)

    return cc_weights, steps


def integrate(x0, xT, nb_steps, integrand, h, parallel=False):
    # Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)
    b_size, x_dim = x0.shape
    h_dim = h.shape[1]
    if parallel:
        x0_t = np.broadcast_to(np.expand_dims(x0, 1), (b_size, nb_steps + 1, x_dim))
        xT_t = np.broadcast_to(np.expand_dims(xT, 1), (b_size, nb_steps + 1, x_dim))
        h_steps = np.broadcast_to(np.expand_dims(h, 1), (b_size, nb_steps + 1, h_dim))
        steps_t = np.broadcast_to(np.expand_dims(steps, 0), (b_size, nb_steps + 1, x_dim))
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.reshape(-1, x0_t.shape[2])
        h_steps = h_steps.reshape(-1, h.shape[1])
        dzs = integrand(X_steps, h_steps)
        dzs = dzs.reshape(xT_t.shape[0], nb_steps + 1, -1)
        dzs = dzs * np.broadcast_to(np.expand_dims(cc_weights, 0), dzs.shape)
        z = dzs.sum(1)
    else:
        z = 0.
        for i in range(nb_steps + 1):
            x = (x0 + (xT - x0)*(steps[i] + 1)/2)
            #print(x.shape, h.shape)
            #return
            dz = integrand(x, h)
            z = z + cc_weights[i]*dz

    return z*(xT - x0)/2


if __name__ == "__main__":
    print("Small check on normal distribution")
    def normal_pdf(x, h):
        mu, sigma = h[:, [0]], h[:, [1]]

        pdf = 1/(np.sqrt(2*math.pi)*sigma)*np.exp((-(x - mu)**2)/(2*sigma**2))
        return pdf

    import matplotlib.pyplot as plt

    x = np.arange(-5, 5, .01).reshape(-1, 1)
    h = np.concatenate((np.zeros_like(x), np.ones_like(x)), 1)

    plt.plot(x, normal_pdf(x, h))
    x0 = np.array([[-1.], [-2.], [-3.]])
    xT = np.array([[1.], [2.], [3.]])
    h = np.array([[0., 1.], [0., 1.], [0., 1.]])
    print(integrate(x0, xT, 10, normal_pdf, h, True))
    print(integrate(x0, xT, 200, normal_pdf, h, False))

    plt.show()