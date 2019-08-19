from models import UMNNMAFFlow
import torch
from lib import dataloader as dl
import lib as transform
import lib.utils as utils
import numpy as np
import os
import pickle
from timeit import default_timer as timer
import torchvision
from tensorboardX import SummaryWriter


writer = SummaryWriter()


def train_mnist(dataset, load=None, gen_image=False, save=None, temperature=.5, real_images=False, nb_iter=5,
                nb_steps=50, solver="CC", hidden_embeding=[1024, 1024, 1024], hidden_derivative=[100, 50, 50, 50, 50],
                embeding_size=30, nb_images=5, conditionnal=False, nb_flow=5, lr=1e-3, weight_decay=1e-2,
                nb_epoch=500, L=1., batch_size=100):
    cuda = 0 if torch.cuda.is_available() else -1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_name = dataset + "/" + str(nb_steps) if save is None else save

    if save is not None or gen_image:
        if not (os.path.isdir(save_name)):
            os.makedirs(save_name)
    logger = utils.get_logger(logpath=os.path.join(save_name, 'logs'), filepath=os.path.abspath(__file__),
                              saving=save is not None)

    cond_in = 10 if conditionnal else 0
    nb_in = 28**2

    model = UMNNMAFFlow(nb_flow=nb_flow, nb_in=nb_in, hidden_derivative=hidden_derivative,
                        hidden_embedding=hidden_embeding, embedding_s=embeding_size, nb_steps=nb_steps, device=device,
                        solver=solver, cond_in=cond_in).to(device)

    if save is not None:
        with open(save + "/model.txt", "w") as f:
            f.write(str(model))

    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if nb_steps > 0:
        max_forward = min(int(3000/(nb_steps/nb_steps * nb_flow * hidden_derivative[0]/100)*784/nb_in), batch_size)
        logger.info("Max forward: %d" % max_forward)
    random_steps = nb_steps <= 0

    if conditionnal:
        train_loader, valid_loader, test_loader = dl.dataloader(dataset, batch_size, cuda=cuda, conditionnal=True)
    else:
        train_loader, valid_loader, test_loader = dl.dataloader(dataset, batch_size, cuda=cuda)

    if load is not None:
        logger.info("Loading model")
        model.load_state_dict(torch.load(load + '/model.pt'))
        model.eval()
        with torch.no_grad():
            # Compute Test loss
            i = 0
            ll_test = 0.
            bpp_avg = 0.
            start = end = timer()
            for batch_idx, (cur_x, target) in enumerate(test_loader):
                if conditionnal:
                    bpp, ll_tmp, z_est = 0, 0, 0
                    for j in range(10):
                        y = target.view(-1, 1)*0 + j
                        y_one_hot = torch.zeros(y.shape[0], 10).scatter(1, y, 1)
                        context = y_one_hot.to(device)
                        bpp_i, ll_tmp_i, z_est_i = model.compute_bpp(cur_x.view(-1, nb_in).to(device), context=context)
                        bpp += bpp_i/10
                        ll_tmp += ll_tmp_i/10
                else:
                    context = None
                    bpp, ll_tmp, z_est = model.compute_bpp(cur_x.view(-1, nb_in).to(device))

                i += 1
                ll_test -= ll_tmp.mean()
                bpp_avg += bpp.mean()
                if i == 5 and nb_epoch > 0:
                    break
                end = timer()
                logger.info("{:d} :Test loss: {:4f} - BPP: {:4f} - Elapsed time per epoch {:4f}".format(
                    i, -ll_test.detach().cpu().item()/i,  -bpp_avg.detach().cpu().item() / i, end - start))

            logger.info("{:d} :Test loss: {:4f} - BPP: {:4f} - Elapsed time per epoch {:4f}".format(
                i, -ll_test.detach().cpu().item() / i, -bpp_avg.detach().cpu().item() / i, end - start))
            nb_sample = nb_images

            # Generate and save images
            if gen_image:
                if real_images:
                    logger.info("Regenerate real images")
                    x, y = next(iter(test_loader))
                    y = y.view(-1, 1)
                    context = torch.zeros(y.shape[0], 10).scatter(1, y, 1).to(device) if conditionnal else None
                    nb_sample = 100
                    z = torch.distributions.Normal(0., 1.).sample(torch.Size([nb_sample, nb_in])).to(
                        device) * torch.arange(0.1, 1.1, .1).unsqueeze(0).expand(int(nb_sample / 10), -1).transpose(0,
                                                                                                                    1) \
                                 .contiguous().view(-1).unsqueeze(1).expand(-1, 784).to(device)
                else:
                    logger.info("Generate random images")
                    z = torch.distributions.Normal(0., 1.).sample(torch.Size([nb_sample, nb_in])).to(
                        device) * temperature

                z_true = z[:nb_sample, :]
                if conditionnal:
                    if real_images:
                        nb_sample = 100
                        z_true = torch.distributions.Normal(0., 1.).sample(torch.Size([nb_sample, nb_in])).to(
                        device) * torch.arange(0.1, 1.1, .1).unsqueeze(0).expand(int(nb_sample/10), -1).transpose(0, 1)\
                            .contiguous().view(-1).unsqueeze(1).expand(-1, 784).to(device)
                        digit = (torch.arange(nb_sample) % 10).float().view(-1, 1)
                    else:
                        digit = (torch.arange(nb_sample) % 10).float().view(-1, 1)
                        logger.info("Creation of: " + str(digit))
                    context = torch.zeros(digit.shape[0], 10).scatter(1, digit.long(), 1).to(device)

                x_est = model.invert(z_true, nb_iter, context=context)

                bpp, ll, _ = model.compute_bpp(x_est, context=context)
                logger.info("Bpp of generated data is: {:4f}".format(bpp.mean().item()))
                logger.info("ll of generated data is: {:4f}".format(ll.mean().item()))

                x = transform.logit_back(x_est.detach().cpu(), 1e-6).view(x_est.shape[0], 1, 28, 28)

                torchvision.utils.save_image(x, save_name + '/' + str(temperature) + 'images.png', nrow=10,
                                             padding=1)
                exit()
        with open(load + '/losses.pkl', 'rb') as f:
            losses_train, losses_test = pickle.load(f)
            cur_epoch = len(losses_test)
    else:
        losses_train = []
        losses_test = []
        cur_epoch = 0
    for epoch in range(cur_epoch, cur_epoch + nb_epoch):
        ll_tot = 0
        i = 0
        start = timer()
        for batch_idx, (cur_x, target) in enumerate(train_loader):
            if conditionnal:
                y = target.view(-1, 1)
                y_one_hot = torch.zeros(y.shape[0], 10).scatter(1, y, 1)
                context = y_one_hot.to(device)
            else:
                cur_x = cur_x.view(-1, nb_in).to(device)
                context = None
            if random_steps:
                nb_steps = np.random.randint(5, 50)*2
                max_forward = min(int(1500 / nb_steps), batch_size)
                model.set_steps_nb(nb_steps)

            cur_x = cur_x.to(device)
            ll = 0.
            opt.zero_grad()
            for cur_su_batch in range(0, batch_size, max_forward):
                ll, z = model.compute_ll(cur_x.view(-1, nb_in)[cur_su_batch:cur_su_batch+max_forward], context=context)
                ll = -ll.mean()/(batch_size/z.shape[0])
                ll.backward()
                ll_tot += ll.detach()
            opt.step()
            if L > 0:
                model.forceLipshitz(L)
            i += 1

            if i % 10 == 0:
                time_tot = timer()
                logger.info("{:d} cur_loss - {:4f} - Average time elapsed per batch {:4f}".format(
                    i, ll_tot.item() / i, (time_tot - start) / i))
                if save:
                    torch.save(model.state_dict(), save_name + '/model.pt')

        ll_tot /= i
        losses_train.append(ll_tot.detach().cpu())

        with torch.no_grad():

            # Generate and save images
            if gen_image and epoch % 10 == 0:
                z = torch.distributions.Normal(0., 1.).sample(torch.Size([nb_images, nb_in])).to(device)*temperature
                if conditionnal:
                    digit = (torch.arange(nb_sample) % 10).float().view(-1, 1)
                    logger.info("Creation of: " + str(digit))
                    context = torch.zeros(digit.shape[0], 10).scatter(1, digit.long(), 1).to(device)

                x = model.invert(z, nb_iter, context=context)
                logger.info("Inversion error: {:4f}".format(torch.abs(z - model.forward(x, context=context)).mean().item()))

                x = x.detach().cpu()
                x = transform.logit_back(x, 1e-6).view(x.shape[0], 1, 28, 28)
                writer.add_image('data/images', torchvision.utils.make_grid(x, nrow=4), epoch)
                torchvision.utils.save_image(x, save_name + '/epoch_{:04d}.png'.format(epoch), nrow=4, padding=1)
            model.set_steps_nb(nb_steps)
            # Compute Test loss
            i = 0
            ll_test = 0.
            for batch_idx, (cur_x, target) in enumerate(valid_loader):
                if conditionnal:
                    y = target.view(-1, 1)
                    y_one_hot = torch.zeros(y.shape[0], 10).scatter(1, y, 1)
                    context = y_one_hot.to(device)
                else:
                    context = None
                ll_tmp, _ = model.compute_ll(cur_x.view(-1, nb_in).to(device), context=context)
                i += 1
                ll_test -= ll_tmp.mean()
            ll_test /= i
        losses_test.append(ll_test.detach().cpu())
        writer.add_scalars('data/' + save_name + "/losses", {"Valid": ll_test.detach().cpu().item(),
                                                             "Train": ll_tot.detach().cpu().item()}, epoch)
        # Save losses
        if save:
            if epoch % 5 == 0:
                if not (os.path.isdir(save_name + '/models')):
                    os.makedirs(save_name + '/models')
                torch.save(model.state_dict(), save_name + '/models/model_{:04d}.pt'.format(epoch))
            with open(save_name + '/losses.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([losses_train, losses_test], f)

        logger.info("epoch: {:d} - Train loss: {:4f} - Test loss: {:4f} - L: {:4f}".format(
            epoch, ll_tot.detach().cpu().item(), ll_test.detach().cpu().item(), model.computeLipshitz(10).detach()))


import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument("-load", default=None, help="where to load")
parser.add_argument("-gen", default=False, action="store_true", help="where to store results")
parser.add_argument("-save", default=None, help="where to store results")
parser.add_argument("-steps", default=50, type=int, help="number of integration steps")
parser.add_argument("-temperature", default=.5, type=float, help="Temperature for sample")
parser.add_argument("-solver", default="CC", help="Temperature for sample")
parser.add_argument("-hidden_embedding", nargs='+', type=int, default=[1024, 1024, 1024], help="Nb neurons for emebding")
parser.add_argument("-hidden_derivative", nargs='+', type=int, default=[100, 50, 50, 50, 50], help="Nb neurons for derivative")
parser.add_argument("-embedding_size", type=int, default=30, help="Size of embedding part")
parser.add_argument("-real_images", type=bool, default=False, help="Generate real images")
parser.add_argument("-dataset", type=str, default="MNIST", help="Dataset")
parser.add_argument("-nb_images", type=int, default=5, help="Number of images to be generated")
parser.add_argument("-conditionnal", type=bool, default=False, help="Conditionning on class or not")
parser.add_argument("-nb_flow", type=int, default=5, help="Number of nets in the flow")
parser.add_argument("-weight_decay", type=float, default=1e-2, help="Weight Decay")
parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("-nb_epoch", type=int, default=500, help="Number of epoch")
parser.add_argument("-nb_iter", type=int, default=500, help="Number of iter for inversion")
parser.add_argument("-Lipshitz", type=float, default=0, help="Lipshitz constant max of linear layer in derivative net")
parser.add_argument("-b_size", type=int, default=100, help="Number of samples per batch")
args = parser.parse_args()

dataset = args.dataset

dir_save = None if args.save is None else dataset + "/" + args.save
dir_load = None if args.load is None else dataset + "/" + args.load

train_mnist(dataset=dataset, load=dir_load, gen_image=args.gen, save=dir_save, nb_steps=args.steps,
            temperature=args.temperature, solver=args.solver, hidden_embeding=args.hidden_embedding,
            hidden_derivative=args.hidden_derivative, real_images=args.real_images, nb_images=args.nb_images,
            conditionnal=args.conditionnal, nb_flow=args.nb_flow, weight_decay=args.weight_decay, lr=args.lr,
            nb_epoch=args.nb_epoch, L=args.Lipshitz, nb_iter=args.nb_iter,
            batch_size=args.b_size)
