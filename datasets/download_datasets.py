# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:58:53 2017
@author: Chin-Wei
# some code adapted from https://github.com/yburda/iwae/blob/master/download_mnist.py

LSUN
https://github.com/fyu/lsun
"""

import urllib
import pickle
import os
import struct
import numpy as np
import gzip
import time
import urllib.request

savedir = 'datasets/data'
mnist = True
cifar10 = False
omniglot = False
maf = False


class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        '''
            @param target: total number of steps expected
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if type(self.sum_values[k]) is list:
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


# mnist
def load_mnist_images_np(imgs_filename):
    with open(imgs_filename, 'rb') as f:
        f.seek(4)
        nimages, rows, cols = struct.unpack('>iii', f.read(12))
        dim = rows * cols
        images = np.fromfile(f, dtype=np.dtype(np.ubyte))
        images = (images / 255.0).astype('float32').reshape((nimages, dim))

    return images


# cifar10
from six.moves.urllib.request import FancyURLopener
import tarfile
import sys


class ParanoidURLopener(FancyURLopener):
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise Exception('URL fetch failure on {}: {} -- {}'.format(url, errcode, errmsg))


def get_file(fname, origin, untar=False):
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print('Downloading data from', origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

        ParanoidURLopener().retrieve(origin, fpath, dl_progress)
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            tfile.extractall(path=datadir)
            tfile.close()
        return untar_fpath

    return fpath


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding="bytes")
        # decode utf8
        for k, v in d.items():
            del (d[str(k)])
            d[str(k)] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_cifar10():
    dirname = "cifar-10-batches-py"
    origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(dirname, origin=origin, untar=True)
    print(path)
    nb_train_samples = 50000

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        print(fpath)
        data, labels = load_batch(fpath)
        X_train[(i - 1) * 10000:i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000:i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if mnist:
        print('dynamically binarized mnist')
        mnist_filenames = ['train-images-idx3-ubyte', 't10k-images-idx3-ubyte']

        for filename in mnist_filenames:
            local_filename = os.path.join(savedir, filename)
            urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/{}.gz".format(filename),
                                       local_filename + '.gz')
            with gzip.open(local_filename + '.gz', 'rb') as f:
                file_content = f.read()
            with open(local_filename, 'wb') as f:
                f.write(file_content)
            np.savetxt(local_filename, load_mnist_images_np(local_filename))
            os.remove(local_filename + '.gz')

        print('statically binarized mnist')
        subdatasets = ['train', 'valid', 'test']
        for subdataset in subdatasets:
            filename = 'binarized_mnist_{}.amat'.format(subdataset)
            url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(
                subdataset)
            local_filename = os.path.join(savedir, filename)
            urllib.request.urlretrieve(url, local_filename)

    if cifar10:
        (X_train, y_train), (X_test, y_test) = load_cifar10()
        pickle.dump((X_train, y_train, X_test, y_test),
                    open('{}/cifar10.pkl'.format(savedir), 'w'))

    if omniglot:
        url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'
        filename = 'omniglot.amat'
        local_filename = os.path.join(savedir, filename)
        urllib.request.urlretrieve(url, local_filename)

    if maf:
        savedir = 'datasets'
        url = 'https://zenodo.org/record/1161203/files/data.tar.gz'
        local_filename = os.path.join(savedir, 'data.tar.gz')
        urllib.request.urlretrieve(url, local_filename)

        tar = tarfile.open(local_filename, "r:gz")
        tar.extractall(savedir)
        tar.close()
        os.remove(local_filename)

