from models import UMNNMAFFlow
import torch
import numpy as np
import os
import pickle
import lib.utils as utils
import datasets
from timeit import default_timer as timer

from tensorboardX import SummaryWriter
writer = SummaryWriter()


def batch_iter(X, batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


def load_data(name):

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def train_uci(dataset, load=None, test=False, save=None, nb_steps=50, solver="CC", hidden_embeding=[300, 300, 300, 300],
              hidden_derivative=[100, 50, 50, 50, 50], embeding_size=30, nb_flow=5, lr=1e-3, weight_decay=1e-2,
              nb_epoch=500, L=1., batch_size = 100, scheduler_rate=.99, scheduler_patience=500, optim="adam"):
    cuda = 0 if torch.cuda.is_available() else -1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_name = "ExperimentsResults/UCIExperiments/" + dataset + "/" + str(nb_steps) if save is None else save
    logger = utils.get_logger(logpath=os.path.join(save_name, 'logs'), filepath=os.path.abspath(__file__), saving=save is not None)

    logger.info("Loading data...")
    data = load_data(dataset)
    data.trn.x = torch.from_numpy(data.trn.x).to(device)
    nb_in = data.trn.x.shape[1]
    data.val.x = torch.from_numpy(data.val.x).to(device)
    data.tst.x = torch.from_numpy(data.tst.x).to(device)
    logger.info("Data loaded.")

    logger.info("Creating model...")
    model = UMNNMAFFlow(nb_flow=nb_flow, nb_in=nb_in, hidden_derivative=hidden_derivative,
                        hidden_embedding=hidden_embeding, embedding_s=embeding_size, nb_steps=nb_steps, device=device,
                        solver=solver).to(device)

    logger.info("Model created.")

    if save is not None:
        with open(save + "/model.txt", "w") as f:
            f.write(str(model))

    if optim == "adam":
        opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    elif optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay, momentum=.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=scheduler_rate, patience=scheduler_patience,
                                                           threshold=1e-2)

    random_steps = nb_steps <= 0

    if load is not None:
        logger.info("Loading model...")
        if cuda >= 0:
            model.load_state_dict(torch.load(load + '/model_best_train.pt'))
        else:
            model.load_state_dict(torch.load(load + '/model_best_train.pt', map_location='cpu'))
        logger.info("Model loaded.")
        if test:
            model.eval()
            with torch.no_grad():
                # Compute Test loss
                i = 0
                ll_test = 0.
                if random_steps:
                    model.set_steps_nb(100)
                for cur_x in batch_iter(data.tst.x, shuffle=True, batch_size=batch_size):

                    ll_tmp, z = model.compute_ll(cur_x)
                    i += 1
                    ll_test -= ll_tmp.mean()
                    logger.info("Test loss: {:4f}".format(ll_test.detach().cpu().data/i))
                ll_test /= i
                logger.info("Number of parameters: {:d} - Test loss: {:4f}".format(len(_flatten(model.parameters())),
                                                                                   ll_test.detach().cpu().data))

        with open(load + '/losses.pkl', 'rb') as f:
            losses_train, losses_test = pickle.load(f)
            cur_epoch = len(losses_test)
    else:
        losses_train = []
        losses_test = []
        cur_epoch = 0
    best_valid = np.inf
    best_train = np.inf
    for epoch in range(cur_epoch, cur_epoch + nb_epoch):
        ll_tot = 0
        i = 0
        start = timer()
        for cur_x in batch_iter(data.trn.x, shuffle=True, batch_size=batch_size):
            if random_steps:
                nb_steps = np.random.randint(5, 50)*2
                model.set_steps_nb(nb_steps)

            opt.zero_grad()
            #Useful to split batch into smaller sub-batches
            max_forward = batch_size
            for cur_su_batch in range(0, batch_size, max_forward):
                ll, z = model.compute_ll(cur_x.view(-1, nb_in)[cur_su_batch:cur_su_batch+max_forward])
                ll = -ll.mean()/(batch_size/max_forward)
                ll.backward()
                ll_tot += ll.detach()

            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1.)
            opt.step()
            if L > 0:
                model.forcei_lpschitz(L)
            i += 1

            if i % 100 == 0:
                time_tot = timer()
                logger.info("{:d} cur_loss {:4f} - Average time elapsed per batch {:4f}".format(i, ll_tot / i, (time_tot-start)/i))
                if save:
                    torch.save(model.state_dict(), save_name + '/model.pt')

        ll_tot /= i
        time_tot = timer()
        losses_train.append(ll_tot.detach().cpu())
        with torch.no_grad():

            # Compute Test loss
            i = 0
            ll_val = 0.
            for cur_x in batch_iter(data.val.x, shuffle=True, batch_size=batch_size):
                ll_tmp, _ = model.computell(cur_x.view(-1, nb_in).to(device))
                i += 1
                ll_val -= ll_tmp.mean()
            ll_val /= i
        losses_test.append(ll_val.detach().cpu())
        writer.add_scalars('data/' + save_name + "/losses", {"Valid": ll_val.detach().cpu().item(),
                                                             "Train": ll_tot.detach().cpu().item()}, epoch)
        scheduler.step(ll_val)
        if ll_val.detach().cpu().item() < best_valid:
            best_valid = ll_val.detach().cpu().item()
            torch.save(model.state_dict(), save_name + '/model_best_valid.pt'.format(epoch))
            if ll_tot.detach().cpu().item() < best_train:
                torch.save(model.state_dict(), save_name + '/model_best_train_valid.pt'.format(epoch))

        if ll_tot.detach().cpu().item() < best_train:
            best_train = ll_tot.detach().cpu().item()
            torch.save(model.state_dict(), save_name + '/model_best_train.pt'.format(epoch))

        # Save losses
        if save:
            if epoch % 5 == 0:
                if not (os.path.isdir(save_name + '/models')):
                    os.makedirs(save_name + '/models')
                torch.save(model.state_dict(), save_name + '/models/model_{:04d}.pt'.format(epoch))
            with open(save_name + '/losses.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([losses_train, losses_test], f)

        logger.info("epoch: {:d} - Train loss: {:4f} - Valid loss: {:4f} - Time elapsed per epoch {:4f}".format(
            epoch, ll_tot.detach().cpu().item(), ll_val.detach().cpu().item(), time_tot-start))


import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument("-load", default=None, help="where to load")
parser.add_argument("-test", default=False, action="store_true", help="Only test")
parser.add_argument("-save", default=None, help="where to store results")
parser.add_argument("-steps", default=50, type=int, help="number of integration steps")
parser.add_argument("-solver", choices=["CC", "CCParallel"], default="CC", help="Solver to use")
parser.add_argument("-hidden_embedding", nargs='+', type=int, default=[512, 512], help="Nb neurons for emebding")
parser.add_argument("-hidden_derivative", nargs='+', type=int, default=[50, 50, 50, 50], help="Nb neurons for derivative")
parser.add_argument("-embedding_size", type=int, default=30, help="Size of embedding part")
parser.add_argument("-nb_flow", type=int, default=5, help="Number of nets in the flow")
parser.add_argument("-weight_decay", type=float, default=1e-2, help="Weight Decay")
parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("-s_rate", type=float, default=.5, help="LR Scheduling rate")
parser.add_argument("-nb_epoch", type=int, default=500, help="Number of epoch")
parser.add_argument("-b_size", type=int, default=500, help="Number of samples per batch")
parser.add_argument("-s_patience", type=int, default=5, help="Number of epoch with no improvement for lr scheduling")
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='miniboone'
)
parser.add_argument("-Lipshitz", type=float, default=0, help="Lipshitz constant max of linear layer in derivative net")
parser.add_argument("-Optim", choices=["adamBNAF", "sgd", "adam"], type=str, default="adam", help="Optimizer")

args = parser.parse_args()

dataset = args.data

dir_save = None if args.save is None else dataset + "/" + args.save
dir_load = None if args.load is None else dataset + "/" + args.load

if dir_save is not None:
    if not (os.path.isdir(dir_save)):
        os.makedirs(dir_save)
    with open(dir_save + "/args.txt", "w") as f:
        f.write(str(args))


train_uci(dataset=dataset, load=dir_load, test=args.test, save=dir_save, nb_steps=args.steps, solver=args.solver,
          hidden_embeding=args.hidden_embedding, hidden_derivative=args.hidden_derivative, nb_flow=args.nb_flow,
          weight_decay=args.weight_decay, lr=args.lr, nb_epoch=args.nb_epoch, L=args.Lipshitz, batch_size=args.b_size,
          scheduler_patience=args.s_patience, scheduler_rate=args.s_rate, optim=args.Optim,
          embeding_size=args.embedding_size)
