import torch
from utils.QuadTree import quad_tree_sampling
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import math

def smooth(densities, areas, centers_x, centers_y, npts):
    smoothed_densities = []
    smoothed_centers_x = []
    smoothed_centers_y = []
    for d, a, x, y in zip(densities, areas, centers_x, centers_y):
        n = int(npts * a)
        smoothed_centers_x += (np.random.randn(n) * 2 * np.sqrt(a)/math.pi + x).tolist()
        smoothed_centers_y += (np.random.randn(n) * 2 * np.sqrt(a)/math.pi + y).tolist()
        smoothed_densities += [d] * n

    return smoothed_centers_x, smoothed_centers_y, smoothed_densities
# Draw contours
theta_range = np.array([[-10., 10.], [-10., 10.]])
for i in range(1):
    # 2d
    # TODO Replace by network evaluation
    def ler_f(t):
        t[:, 0] = 1/t[:, 0]
        return torch.ones(t.shape[0])

    # TODO Replace by correct priors
    def prior_f_2d(t):
        p = 0.5 * 1/(2*math.pi)*np.exp(-.5*(((t[:, 0]) - 5.)**2 + t[:, 1]**2)) + 0.5* 1 / (2 * math.pi) * np.exp(-.5 * ((t[:, 0] + 5) ** 2 + (t[:, 1]) ** 2))
        return p

    def posterior_f_2d(x, y):
        t = torch.cat((x.unsqueeze(1), y.unsqueeze(1)), 1)
        return prior_f_2d(t) * ler_f(t)

    (samples_x, samples_y, samples_cumulative, sample_area), _ = quad_tree_sampling(theta_range[0, :], theta_range[1, :], 20, 25, posterior_f_2d)
    print(np.array(samples_cumulative).sum(), len(samples_cumulative), len(sample_area))
    plt.figure()
    print(np.array(samples_cumulative).shape)
    plt.scatter(samples_x, samples_y, c=np.log(np.array(samples_cumulative) / np.array(sample_area)),
                s=np.array(sample_area) * 500, alpha=.95, edgecolors='none')
    plt.colorbar()
    samples_x, samples_y, samples_cumulative, sample_area = np.array(samples_x), np.array(samples_y), np.array(samples_cumulative), np.array(sample_area)
    order = np.argsort(samples_cumulative / sample_area)[::-1]
    samples_x, samples_y, samples_cumulative = samples_x[order], samples_y[order], samples_cumulative[order]
    samples_F = np.cumsum(samples_cumulative)
    print(samples_x.shape, samples_y.shape, samples_F.shape)
    cp = plt.tricontour(samples_x, samples_y, samples_F, levels=[.5, .7, .9], cmap=cm.Greys_r)
    plt.clabel(cp, fontsize=10)
    plt.show()

exit()