import pdb
import time
import pickle
import numpy as np

import lasagne
import theano
import theano.tensor as T

from lasagne.layers import *
from layers import GumbelSoftmaxSampleLayer
from distributions import log_bernoulli
from model import Model
from helpers import *


class GSM(Model):
    """Gumbel Softmax w/ categorical latent variables
    https://arxiv.org/pdf/1611.01144v2.pdf

    Epoch 100 of 100 took 60.530s (500 minibatches)
    training loss/acc:		  95.905443	-19.943117
    validation loss/acc:	  98.537678	-19.804097
    """
    def __init__(
        self, n_dim, n_out, n_chan=1, n_superbatch=12800,
        opt_alg='adam', opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}
    ):
        # invoke parent constructor
        # create shared data variables
        train_set_x = theano.shared(
            np.empty(
                (n_superbatch, n_chan*n_dim*n_dim),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        val_set_x = theano.shared(
            np.empty(
                (n_superbatch, n_chan*n_dim*n_dim),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        # create y-variables
        train_set_y = theano.shared(
            np.empty(
                (n_superbatch,),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        val_set_y = theano.shared(
            np.empty(
                (n_superbatch,),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        train_set_y_int = T.cast(train_set_y, 'int32')
        val_set_y_int = T.cast(val_set_y, 'int32')

        # create shared learning variables
        tau = theano.shared(
            np.float32(5.0), name='temperature',
            allow_downcast=True, borrow=False,
        )
        self.tau = tau

        # create input vars
        x = T.matrix(dtype=theano.config.floatX)
        s = T.tensor3(dtype=theano.config.floatX)
        y = T.ivector()
        idx1, idx2 = T.lscalar(), T.lscalar()
        self.inputs = (x, y, idx1, idx2, s)

        # create lasagne model
        self.network = self.create_model(x, y, n_dim, n_out, n_chan)

        # create objectives
        loss, acc = self.create_objectives(deterministic=False)
        self.objectives = (loss, acc)

        # create hallucinations
        sample = self.gen_samples(deterministic=False)
        self.dream = theano.function([s], sample, on_unused_input='warn')

        # create gradients
        grads = self.create_gradients(loss, deterministic=False)

        # get params
        params = self.get_params()

        # create updates
        alpha = T.scalar(dtype=theano.config.floatX)  # learning rate
        updates = self.create_updates(
            grads, params, alpha, opt_alg, opt_params,
        )

        self.train = theano.function(
            [idx1, idx2, alpha], [loss, acc],
            updates=updates,
            givens={
                x: train_set_x[idx1:idx2],
                y: train_set_y_int[idx1:idx2]
            },
            on_unused_input='warn',
        )

        self.loss = theano.function(
            [x, y], [loss, acc],
            on_unused_input='warn',
        )

        # save config
        self.n_dim = n_dim
        self.n_out = n_out
        self.n_chan = n_chan
        self.n_superbatch = n_superbatch
        self.alg = opt_alg

        # save data variables
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.val_set_x = val_set_x
        self.val_set_y = val_set_y
        self.data_loaded = False

        # save neural network
        self.params = params
        self.grads = (grads, None)
        self.metrics = (loss, acc)

    def create_model(self, x, y, n_dim, n_out, n_chan=1):
        n_class = 10  # number of classes
        n_cat = 30  # number of categorical distributions
        n_out = n_dim * n_dim * n_chan
        n_in = n_out
        tau = self.tau

        # create the encoder network
        q_net_in = InputLayer(shape=(None, n_in), input_var=x)
        q_net = DenseLayer(q_net_in, num_units=512, nonlinearity=T.nnet.relu)
        q_net = DenseLayer(q_net, num_units=256, nonlinearity=T.nnet.relu)
        # sample from Gumble-Softmax posterior
        q_net_mu = DenseLayer(q_net, n_cat*n_class, nonlinearity=None)
        q_net_mu = reshape(q_net_mu, (-1, n_class))
        q_net_sample = GumbelSoftmaxSampleLayer(q_net_mu, tau)
        q_net_sample = reshape(q_net_sample, (-1, n_cat, n_class))
        # create the decoder network
        p_net_in = InputLayer(shape=(None, n_cat, n_class))
        p_net = DenseLayer(flatten(p_net_in), 256, nonlinearity=T.nnet.relu)
        p_net = DenseLayer(p_net, 512, nonlinearity=T.nnet.relu)
        p_net_mu = DenseLayer(p_net, n_out, nonlinearity=T.nnet.sigmoid)

        # save network params
        self.n_class = n_class
        self.n_cat = n_cat

        self.input_layers = (q_net_in, p_net_in)

        return q_net_mu, p_net_mu, q_net_sample

    def create_objectives(self, deterministic=False):
        x = self.inputs[0]

        # load network params
        n_class = self.n_class
        n_cat = self.n_cat

        # load network output
        q_net_in, p_net_in = self.input_layers
        q_net_mu, p_net_mu, q_net_sample = self.network
        q_mu, q_sample = get_output([q_net_mu, q_net_sample])
        p_mu = get_output(p_net_mu, {p_net_in : q_sample})

        # define the loss
        q_z = T.nnet.softmax(q_mu)
        log_q_z = T.log(q_z + 1e-20)
        log_p_x = log_bernoulli(x, p_mu)

        kl_tmp = T.reshape(q_z * (log_q_z - T.log(1.0 / n_class)), [-1 , n_cat, n_class])
        KL = T.sum(kl_tmp, axis=[1, 2])
        elbo = T.sum(log_p_x, axis=1) - KL
        loss = T.mean(-elbo)

        return loss, -T.mean(KL)

    def gen_samples(self, deterministic=False):
        s = self.inputs[-1]
        # put it through the decoder
        _, p_net_in = self.input_layers
        _, p_net_mu, _ = self.network
        p_mu = get_output(p_net_mu, {p_net_in : s})

        return p_mu

    def get_params(self):
        q_net_mu, p_net_mu, _ = self.network
        q_params = get_all_params(q_net_mu)
        p_params = get_all_params(p_net_mu)

        return p_params + q_params

    def hallucinate(self):
        """Generate new samples by passing noise into the decoder"""
        # load network params
        size = 100
        n_cat, n_class, n_dim = self.n_cat, self.n_class, self.n_dim
        n_mag = size * n_cat
        img_size = np.sqrt(size)

        # generate noisy inputs
        noise = self.gen_noise(n_mag, n_class)
        noise = np.reshape(noise,[size, n_cat, n_class])

        p_mu = self.dream(noise)
        if p_mu is None: return None
        p_mu = p_mu.reshape((img_size, img_size, n_dim, n_dim))
        # split into img_size (1,img_size,n_dim,n_dim) images,
        # concat along columns -> 1,img_size,n_dim,n_dim*img_size
        p_mu = np.concatenate(np.split(p_mu, img_size, axis=0), axis=3)
        # split into img_size (1,1,n_dim,n_dim*img_size) images,
        # concat along rows -> 1,1,n_dim*img_size,n_dim*img_size
        p_mu = np.concatenate(np.split(p_mu, img_size, axis=1), axis=2)
        return np.squeeze(p_mu)

    def fit(
        self, X_train, Y_train, X_val, Y_val,
        n_epoch=10, n_batch=100, logname='run',
    ):
        """Train the model"""

        alpha = 1.0  # learning rate, which can be adjusted later
        tau0 = 1.0  # initial temp
        MIN_TEMP = 0.5  # minimum temp
        ANNEAL_RATE = 0.00003  # adjusting rate
        np_temp = tau0
        n_data = len(X_train)
        n_superbatch = self.n_superbatch
        i = 1  # track # of train() executions

        # flatten X_train, X_val
        n_flat_dim = np.prod(X_train.shape[1:])
        X_train = X_train.reshape(-1, n_flat_dim)
        X_val = X_val.reshape(-1, n_flat_dim)

        for epoch in range(n_epoch):
            # In each epoch, we do a full pass over the training data:
            train_batches, train_err, train_acc = 0, 0, 0
            start_time = time.time()

            # iterate over superbatches to save time on GPU memory transfer
            for X_sb, Y_sb in self.iterate_superbatches(
                X_train, Y_train, n_superbatch,
                datatype='train', shuffle=True,
            ):
                for idx1, idx2 in iterate_minibatch_idx(len(X_sb), n_batch):
                    err, acc = self.train(idx1, idx2, alpha)

                    # anneal temp and learning rate
                    if i % 1000 == 1:
                        alpha *= 0.9
                        np_temp = np.maximum(tau0*np.exp(-ANNEAL_RATE*i), MIN_TEMP)
                        self.tau.set_value(np_temp, borrow=False)

                    # collect metrics
                    i += 1
                    train_batches += 1
                    train_err += err
                    train_acc += acc
                    if train_batches % 100 == 0:
                        n_total = epoch * n_data + n_batch * train_batches
                        metrics = [n_total, train_err / train_batches, train_acc / train_batches]
                        log_metrics(logname, metrics)

            print "Epoch {} of {} took {:.3f}s ({} minibatches)".format(
                epoch + 1, n_epoch,
                time.time() - start_time,
                train_batches,
            )

            # make a full pass over the training data and record metrics:
            train_err, train_acc = evaluate(self.loss, X_train, Y_train, batchsize=100)
            val_err, val_acc = evaluate(self.loss, X_val, Y_val, batchsize=100)

            print "  training loss/acc:\t\t{:.6f}\t{:.6f}".format(train_err, train_acc)
            print "  validation loss/acc:\t\t{:.6f}\t{:.6f}".format(val_err, val_acc)

            metrics = [ epoch, train_err, train_acc, val_err, val_acc ]
            log_metrics(logname + '.val', metrics)
