from models import UMNNMAFFlow
import torch
import lib.toy_data as toy_data
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
import lib.utils as utils
import lib.visualize_flow as vf

green = '#e15647'
black = '#2d5468'
white_bg = '#ececec'
def summary_plots(x, x_test, folder, epoch, model, ll_tot, ll_test):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(1, 1, 1, aspect="equal")
    vf.plt_flow(model.compute_ll, ax)
    #ax = plt.subplot(1, 3, 2, aspect="equal")
    #vf.plt_samples(toy_data.inf_train_gen(toy, batch_size=50000), ax, npts=500)
    #ax = plt.subplot(1, 3, 3, aspect="equal")
    #samples = model.invert(torch.distributions.Normal(0., 1.).sample([5000, 2]), 8, "Binary")
    #vf.plt_samples(samples.detach().numpy(), ax, title="$x\sim q(x)$")
    plt.savefig("%s/flow_%d.pdf" % (folder + toy, epoch))
    plt.savefig("%s/flow_%d.png" % (folder + toy, epoch))
    plt.close(fig)
    fig = plt.figure()

    z = torch.distributions.Normal(0., 1.).sample(x_test.shape)
    plt.figure(figsize=(7, 7))
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)

    plt.xlabel("$z_1$", fontsize=20)
    plt.ylabel("$z_2$", fontsize=20)
    plt.scatter(z[:, 0], z[:, 1], alpha=.2, color=green)
    x_min = z.min(0)[0] - .5
    x_max = z.max(0)[0] + .5
    ticks = [1, 1]

    plt.xticks([-4, 0, 4])
    plt.yticks([-4, 0, 4])
    #plt.grid(True)
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor(white_bg)

    ax.tick_params(axis='x', colors=black)
    ax.tick_params(axis='y', colors=black)
    ax.spines['bottom'].set_color(black)
    ax.spines['left'].set_color(black)
    #plt.xticks(np.arange(int(x_min[0]), int(x_max[0]), ticks[0]), np.arange(int(x_min[0]), int(x_max[0]), ticks[0]))
    #plt.yticks(np.arange(int(x_min[1]), int(x_max[1]), ticks[1]), np.arange(int(x_min[1]), int(x_max[1]), ticks[1]))
    plt.tight_layout()
    plt.savefig("noise.png", transparent=True)

    z_pred = model.forward(x_test)
    z_pred = z_pred.detach().cpu().numpy()
    #plt.subplot(221)
    plt.figure()
    plt.title("z pred")
    plt.scatter(z_pred[:, 0], z_pred[:, 1], alpha=.2)
    plt.xticks(np.arange(int(x_min[0]), int(x_max[0]), ticks[0]), np.arange(int(x_min[0]), int(x_max[0]), ticks[0]))
    plt.yticks(np.arange(int(x_min[1]), int(x_max[1]), ticks[1]), np.arange(int(x_min[1]), int(x_max[1]), ticks[1]))
    plt.savefig("test2.png")

    start = timer()
    z = torch.distributions.Normal(0., 1.).sample((10000, 2))
    x_pred = model.invert(z, 5, "ParallelSimpler")
    end = timer()
    print("Inversion time: {:4f}s".format(end - start))
    plt.subplot(223)
    #plt.title("x pred")

    x_pred = x_pred.detach().cpu().numpy()
    plt.scatter(x_pred[:, 0], x_pred[:, 1], alpha=.2)
    x_min = x.min(0)[0] - .5
    x_max = x.max(0)[0] + .5
    ticks = [1, 1]
    plt.xticks(np.arange(int(x_min[0]), int(x_max[0]), ticks[0]), np.arange(int(x_min[0]), int(x_max[0]), ticks[0]))
    plt.yticks(np.arange(int(x_min[1]), int(x_max[1]), ticks[1]), np.arange(int(x_min[1]), int(x_max[1]), ticks[1]))

    #plt.subplot(224)
    plt.figure(figsize=(7, 7))
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    #cmap = matplotlib.cm.get_cmap(None)
    #ax.set_facecolor(cmap(0.))
    # ax.invert_yaxis()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xticks([-4, 0, 4])
    plt.yticks([-4, 0, 4])
    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20)
    plt.scatter(x[:, 0], x[:, 1], alpha=.2, color='#e15647')
    #plt.xticks(np.arange(-5, 5.1, 2))
    #plt.yticks(np.arange(-5, 5.1, 2))
    #plt.grid(True)
    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_facecolor(white_bg)

    ax.tick_params(axis='x', colors=black)
    ax.tick_params(axis='y', colors=black)
    ax.spines['bottom'].set_color(black)
    ax.spines['left'].set_color(black)
    #plt.xticks(np.arange(int(x_min[0]), int(x_max[0]), ticks[0]), np.arange(int(x_min[0]), int(x_max[0]), ticks[0]))
    #plt.yticks(np.arange(int(x_min[1]), int(x_max[1]), ticks[1]), np.arange(int(x_min[1]), int(x_max[1]), ticks[1]))
    plt.tight_layout()
    plt.savefig("8gaussians.png", transparent=True)

    plt.suptitle(str(("epoch: ", epoch, "Train loss: ", ll_tot.item(), "Test loss: ", ll_test.item())))
    plt.savefig("%s/%d.png" % (folder + toy, epoch))
    plt.close(fig)


def train_toy(toy, load=True, nb_steps=20, nb_flow=1, folder=""):
    device = "cpu"
    logger = utils.get_logger(logpath=os.path.join(folder, toy, 'logs'), filepath=os.path.abspath(__file__))

    logger.info("Creating model...")
    model = UMNNMAFFlow(nb_flow=nb_flow, nb_in=2, hidden_derivative=[100, 100, 100, 100], hidden_embedding=[100, 100, 100, 100],
                        embedding_s=10, nb_steps=nb_steps, device=device).to(device)
    logger.info("Model created.")
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

    if load:
        logger.info("Loading model...")
        model.load_state_dict(torch.load(folder + toy+'/model.pt'))
        model.train()
        opt.load_state_dict(torch.load(folder + toy+'/ADAM.pt'))
        logger.info("Model loaded.")

    nb_samp = 100
    batch_size = 100

    x_test = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)
    x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=1000)).to(device)

    for epoch in range(10000):
        ll_tot = 0
        start = timer()
        for j in range(0, nb_samp, batch_size):
            cur_x = torch.tensor(toy_data.inf_train_gen(toy, batch_size=batch_size)).to(device)
            ll, z = model.compute_ll(cur_x)
            ll = -ll.mean()
            ll_tot += ll.detach()/(nb_samp/batch_size)
            loss = ll
            opt.zero_grad()
            loss.backward()
            opt.step()
        end = timer()
        ll_test, _ = model.compute_ll(x_test)
        ll_test = -ll_test.mean()
        logger.info("epoch: {:d} - Train loss: {:4f} - Test loss: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                    format(epoch, ll_tot.item(), ll_test.item(), end-start))

        if (epoch % 100) == 0:
            summary_plots(x, x_test, folder, epoch, model, ll_tot, ll_test)
            torch.save(model.state_dict(), folder + toy + '/model.pt')
            torch.save(opt.state_dict(), folder + toy + '/ADAM.pt')


import argparse
datasets = ["8gaussians", "swissroll", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "line-noisy",
            "circles", "joint_gaussian"]

parser = argparse.ArgumentParser(description='')
parser.add_argument("-dataset", default=None, choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
args = parser.parse_args()

if args.dataset is None:
    toys = datasets
else:
    toys = [args.dataset]

for toy in toys:
    if not(os.path.isdir(args.folder + toy)):
        os.makedirs(args.folder + toy)
    train_toy(toy, load=args.load, folder=args.folder)
