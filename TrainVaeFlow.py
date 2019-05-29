# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random

import os

import datetime

import lib.utils as utils

from models.vae_lib.models import VAE
from models.vae_lib.optimization.training import train, evaluate
from models.vae_lib.utils.load_data import load_dataset
from models.vae_lib.utils.plotting import plot_training_curve


from tensorboardX import SummaryWriter
writer = SummaryWriter()

SOLVERS = ["CC", "CCParallel", "Simpson"]
parser = argparse.ArgumentParser(description='PyTorch VAE Normalizing flows')

parser.add_argument(
    '-d', '--dataset', type=str, default='mnist', choices=['mnist', 'freyfaces', 'omniglot', 'caltech'],
    metavar='DATASET', help='Dataset choice.'
)

parser.add_argument(
    '-freys', '--freyseed', type=int, default=123, metavar='FREYSEED',
    help="""Seed for shuffling frey face dataset for test split. Ignored for other datasets.
                    Results in paper are produced with seeds 123, 321, 231"""
)

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument(
    '-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
    help='how many batches to wait before logging training status'
)

parser.add_argument(
    '-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
    help='output directory for model snapshots etc.'
)

# optimization settings
parser.add_argument(
    '-e', '--epochs', type=int, default=2000, metavar='EPOCHS', help='number of epochs to train (default: 2000)'
)
parser.add_argument(
    '-es', '--early_stopping_epochs', type=int, default=35, metavar='EARLY_STOPPING',
    help='number of early stopping epochs'
)

parser.add_argument(
    '-bs', '--batch_size', type=int, default=100, metavar='BATCH_SIZE', help='input batch size for training'
)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, metavar='LEARNING_RATE', help='learning rate')

parser.add_argument(
    '-w', '--warmup', type=int, default=100, metavar='N',
    help='number of epochs for warm-up. Set to 0 to turn warmup off.'
)
parser.add_argument('--max_beta', type=float, default=1., metavar='MB', help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB', help='min beta for warm-up')
parser.add_argument(
    '-f', '--flow', type=str, default='no_flow', choices=[
        'planar', 'iaf', 'householder', 'orthogonal', 'triangular', 'MMAF', 'no_flow'
    ], help="""Type of flows to use, no flows can also be selected"""
)
parser.add_argument('-r', '--rank', type=int, default=1)
parser.add_argument(
    '-nf', '--num_flows', type=int, default=4, metavar='NUM_FLOWS',
    help='Number of flow layers, ignored in absence of flows'
)
parser.add_argument(
    '-nv', '--num_ortho_vecs', type=int, default=8, metavar='NUM_ORTHO_VECS',
    help=""" For orthogonal flow: How orthogonal vectors per flow do you need.
                    Ignored for other flow types."""
)
parser.add_argument(
    '-nh', '--num_householder', type=int, default=8, metavar='NUM_HOUSEHOLDERS',
    help=""" For Householder Sylvester flow: Number of Householder matrices per flow.
                    Ignored for other flow types."""
)
parser.add_argument(
    '-mhs', '--made_h_size', type=int, default=320, metavar='MADEHSIZE',
    help='Width of mades for iaf and MMAF. Ignored for all other flows.'
)
parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE', help='how many stochastic hidden units')
# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

# MMAF settings
parser.add_argument("-steps", default=50, type=int, help="number of integration steps")
parser.add_argument("-solver", default="CC", help="Solver used")
parser.add_argument("-hidden_embedding", nargs='+', type=int, default=[512, 512], help="Nb neurons for emebding")
parser.add_argument("-hidden_derivative", nargs='+', type=int, default=[50, 50, 50, 50], help="Nb neurons for derivative")
parser.add_argument("-embedding_size", type=int, default=30, help="Size of embedding part")
parser.add_argument("-Lipshitz", type=float, default=0, help="Lipshitz constant max of linear layer in derivative net")

# evaluation
parser.add_argument('--evaluate', type=eval, default=False, choices=[True, False])
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--retrain_encoder', type=eval, default=False, choices=[True, False])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)

if args.cuda:
    # gpu device number
    torch.cuda.set_device(args.gpu_num)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


def run(args, kwargs):
    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    snapshots_path = os.path.join(args.out_dir, 'vae_' + args.dataset + '_')
    snap_dir = snapshots_path + args.flow

    if args.flow != 'no_flow':
        snap_dir += '_' + 'num_flows_' + str(args.num_flows)

    if args.flow == 'orthogonal':
        snap_dir = snap_dir + '_num_vectors_' + str(args.num_ortho_vecs)
    elif args.flow == 'orthogonalH':
        snap_dir = snap_dir + '_num_householder_' + str(args.num_householder)
    elif args.flow == 'iaf':
        snap_dir = snap_dir + '_madehsize_' + str(args.made_h_size)
    elif args.flow == "MMAF":
        snap_dir = snap_dir + 'MMAF'
    elif args.flow == 'permutation':
        snap_dir = snap_dir + '_' + 'kernelsize_' + str(args.kernel_size)
    elif args.flow == 'mixed':
        snap_dir = snap_dir + '_' + 'num_householder_' + str(args.num_householder)

    if args.retrain_encoder:
        snap_dir = snap_dir + '_retrain-encoder_'
    elif args.evaluate:
        snap_dir = snap_dir + '_evaluate_'

    snap_dir = snap_dir + '__' + args.model_signature + '/'

    args.snap_dir = snap_dir

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    # logger
    utils.makedirs(args.snap_dir)
    logger = utils.get_logger(logpath=os.path.join(args.snap_dir, 'logs'), filepath=os.path.abspath(__file__))

    logger.info(args)

    # SAVING
    torch.save(args, snap_dir + args.flow + '.config')

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    if not args.evaluate:

        # ==============================================================================================================
        # SELECT MODEL
        # ==============================================================================================================
        # flow parameters and architecture choice are passed on to model through args

        if args.flow == 'no_flow':
            model = VAE.VAE(args)
        elif args.flow == 'planar':
            model = VAE.PlanarVAE(args)
        elif args.flow == 'iaf':
            model = VAE.IAFVAE(args)
        elif args.flow == 'orthogonal':
            model = VAE.OrthogonalSylvesterVAE(args)
        elif args.flow == 'householder':
            model = VAE.HouseholderSylvesterVAE(args)
        elif args.flow == 'triangular':
            model = VAE.TriangularSylvesterVAE(args)
        elif args.flow == 'MMAF':
            model = VAE.MMAVAE(args)
        else:
            raise ValueError('Invalid flow choice')

        if args.retrain_encoder:
            logger.info(f"Initializing decoder from {args.model_path}")
            dec_model = torch.load(args.model_path)
            dec_sd = {}
            for k, v in dec_model.state_dict().items():
                if 'p_x' in k:
                    dec_sd[k] = v
            model.load_state_dict(dec_sd, strict=False)

        if args.cuda:
            logger.info("Model on GPU")
            model.cuda()

        logger.info(model)

        if args.retrain_encoder:
            parameters = []
            logger.info('Optimizing over:')
            for name, param in model.named_parameters():
                if 'p_x' not in name:
                    logger.info(name)
                    parameters.append(param)
        else:
            parameters = model.parameters()

        optimizer = optim.Adamax(parameters, lr=args.learning_rate, eps=1.e-7)

        # ==================================================================================================================
        # TRAINING
        # ==================================================================================================================
        train_loss = []
        val_loss = []

        # for early stopping
        best_loss = np.inf
        best_bpd = np.inf
        e = 0
        epoch = 0

        train_times = []

        for epoch in range(1, args.epochs + 1):

            t_start = time.time()
            tr_loss = train(epoch, train_loader, model, optimizer, args, logger)
            train_loss.append(tr_loss)
            train_times.append(time.time() - t_start)
            logger.info('One training epoch took %.2f seconds' % (time.time() - t_start))

            v_loss, v_bpd = evaluate(val_loader, model, args, logger, epoch=epoch)

            val_loss.append(v_loss)

            writer.add_scalars('data/' + args.snap_dir + "/losses", {"Valid": v_loss,
                                                                 "Train": tr_loss.sum() / len(train_loader)}, epoch)

            # early-stopping
            if v_loss < best_loss:
                e = 0
                best_loss = v_loss
                if args.input_type != 'binary':
                    best_bpd = v_bpd
                logger.info('->model saved<-')
                torch.save(model, snap_dir + args.flow + '.model')
                # torch.save(model, snap_dir + args.flow + '_' + args.architecture + '.model')

            elif (args.early_stopping_epochs > 0) and (epoch >= args.warmup):
                e += 1
                if e > args.early_stopping_epochs:
                    break

            if args.input_type == 'binary':
                logger.info(
                    '--> Early stopping: {}/{} (BEST: loss {:.4f})\n'.format(e, args.early_stopping_epochs, best_loss)
                )

            else:
                logger.info(
                    '--> Early stopping: {}/{} (BEST: loss {:.4f}, bpd {:.4f})\n'.
                    format(e, args.early_stopping_epochs, best_loss, best_bpd)
                )

            if math.isnan(v_loss):
                raise ValueError('NaN encountered!')

        train_loss = np.hstack(train_loss)
        val_loss = np.array(val_loss)

        plot_training_curve(train_loss, val_loss, fname=snap_dir + '/training_curve_%s.pdf' % args.flow)

        # training time per epoch
        train_times = np.array(train_times)
        mean_train_time = np.mean(train_times)
        std_train_time = np.std(train_times, ddof=1)
        logger.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

        # ==================================================================================================================
        # EVALUATION
        # ==================================================================================================================

        logger.info(args)
        logger.info('Stopped after %d epochs' % epoch)
        logger.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

        final_model = torch.load(snap_dir + args.flow + '.model')
        validation_loss, validation_bpd = evaluate(val_loader, final_model, args, logger)

    else:
        validation_loss = "N/A"
        validation_bpd = "N/A"
        logger.info(f"Loading model from {args.model_path}")
        final_model = torch.load(args.model_path)

    test_loss, test_bpd = evaluate(test_loader, final_model, args, logger, testing=False)
    logger.info('FINAL EVALUATION ON VALIDATION SET. ELBO (VAL): {:.4f}'.format(validation_loss))
    logger.info('FINAL EVALUATION ON TESTING SET. ELBO (VAL): {:.4f}'.format(test_loss))
    if args.input_type != 'binary':
        logger.info('FINAL EVALUATION ON VALIDATION SET. ELBO (VAL) BPD : {:.4f}'.format(validation_bpd))
        logger.info('FINAL EVALUATION ON TEST SET. NLL (TEST) BPD: {:.4f}'.format(test_bpd))
    return

    logger.info('FINAL EVALUATION ON VALIDATION SET. ELBO (VAL): {:.4f}'.format(validation_loss))
    logger.info('FINAL EVALUATION ON TEST SET. NLL (TEST): {:.4f}'.format(test_loss))
    if args.input_type != 'binary':
        logger.info('FINAL EVALUATION ON VALIDATION SET. ELBO (VAL) BPD : {:.4f}'.format(validation_bpd))
        logger.info('FINAL EVALUATION ON TEST SET. NLL (TEST) BPD: {:.4f}'.format(test_bpd))


if __name__ == "__main__":

    run(args, kwargs)
