import pdb
import time, timeit
import pickle
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from model import Model
from helpers import *

theano.config.optmizer = 'None'

class RBM(Model):
    """Restricted Boltzmann Machine
    RBM code adapted from http://deeplearning.net/tutorial/rbm.html

    Epoch 15 of 15 took 168.265s (2500 minibatches)
        training loss/acc:		-62.755739271	None

    Training Params
    ---------------
    batch_size: 20
    learning_rate: 0.1
    """
    def __init__(
        self, n_dim, n_out, n_chan=1, n_superbatch=12800, opt_alg='adam',
        opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}
    ):
        """RBM constructor.
        Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.
        """
        self.numpy_rng = np.random.RandomState(1234)
        self.theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 30))

        # extract training params
        lr = opt_params.get('lr')
        n_batch = opt_params.get('nb')

        # create shared objects for x and y
        train_set_x = theano.shared(
            np.empty(
                (n_superbatch, n_chan, n_dim, n_dim),
                dtype=theano.config.floatX
            ), borrow=False,
        )
        val_set_x = theano.shared(np.empty(
            (n_superbatch, n_chan, n_dim, n_dim),
            dtype=theano.config.floatX),
            borrow=False,
        )

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

        # allocate symbolic variables for the data
        X = T.tensor4(dtype=theano.config.floatX)
        S = T.tensor3(dtype=theano.config.floatX)
        Y = T.ivector()
        idx1, idx2 = T.lscalar(), T.lscalar()
        alpha = T.scalar(dtype=theano.config.floatX)  # adjustable learning rate
        self.inputs = (X, Y, idx1, idx2, S)

        # create model
        self.network = self.create_model(n_dim, n_out, n_chan)

        # initialize storage for the persistent chain (state = hidden
        # layer of chain)
        persistent_chain = theano.shared(
            np.zeros(
                (n_batch, self.n_hidden),
                dtype=theano.config.floatX
            ), borrow=True,
        )

        # get the cost and the gradient corresponding to one step of CD-15
        cost, acc, updates = self.get_cost_updates(
            X, alpha, lr=lr, persistent=persistent_chain,
        )
        self.objectives = (cost, acc)

        self.train = theano.function(
            [idx1, idx2, alpha],
            [cost, acc],
            updates=updates,
            givens={X : train_set_x[idx1:idx2], Y : train_set_y_int[idx1:idx2]},
            on_unused_input='warn',
        )

        self.n_batch = n_batch
        self.loss = theano.function([X, Y], [cost, acc], on_unused_input='warn')

        # save config
        self.n_dim = n_dim
        self.n_out = n_out
        self.n_superbatch = n_superbatch
        self.alg = opt_alg

        # save data variables
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.val_set_x = val_set_x
        self.val_set_y = val_set_y
        self.data_loaded = False

        # save neural network
        self.params = self.get_params()
        self.metrics = (cost, acc)

    def create_model(self, n_dim, n_out, n_chan=1):
        n_visible = n_chan*n_dim*n_dim  # size of visible layer
        n_hidden  = 500  # size of hidden layer
        k_steps   = 15  # number of steps during CD/PCD

        # W is initialized with `initial_W` which is uniformely
        # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
        # converted using asarray to dtype theano.config.floatX so
        # that the code is runable on GPU
        initial_W = np.asarray(
            self.numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)
            ), dtype=theano.config.floatX,
        )
        # theano shared variables for weights and biases
        W = theano.shared(value=initial_W, name='W', borrow=True)

        # create shared variable for hidden units bias
        hbias = theano.shared(
            value=np.zeros(
                n_hidden,
                dtype=theano.config.floatX
            ), name='hbias',
            borrow=True,
        )

        # create shared variable for visible units bias
        vbias = theano.shared(
            value=np.zeros(
                n_visible,
                dtype=theano.config.floatX
            ), name='vbias',
            borrow=True,
        )

        # the data is presented as rasterized images
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

        # network params
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k_steps = k_steps

        return None

    def free_energy(self, v_sample):
        """Function to compute the free energy"""
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        """This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        """
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        """This function infers state of hidden units given visible units"""
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(
            size=h1_mean.shape,
            n=1, p=h1_mean,
            dtype=theano.config.floatX,
        )
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        """
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """This function infers state of visible units given hidden units"""
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(
            size=v1_mean.shape,
            n=1, p=v1_mean,
            dtype=theano.config.floatX,
        )
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """This function implements one step of Gibbs sampling,
        starting from the hidden state
        """
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """This function implements one step of Gibbs sampling,
        starting from the visible state
        """
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_params(self):
        return self.params

    def get_cost_updates(self, X, alpha, lr=0.1, persistent=None):
        """Returns the updates dictionary.
        The dictionary contains the update rules
        for weights and biases but also an update of the shared variable used to
        store the persistent chain, if one is used.

        :param lr: learning rate used to train the RBM
        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.
        """
        x = X.flatten(2)

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(x)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        chain_start = ph_sample if persistent is None else persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=15,
            name="gibbs_hvh"
        )

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(x)) - T.mean(self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        gparams = [grad * alpha for grad in gparams]

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(x, updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(
                x, updates, pre_sigmoid_nvs[-1])

        # TODO fill in real accuracies
        return monitoring_cost, monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, X, updates):
        """Stochastic approximation to the pseudo-likelihood"""
        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(X)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, X, updates, pre_sigmoid_nv):
        """
        Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.
        """

        cross_entropy = T.mean(
            T.sum(
                X * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - X) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=10, n_batch=100, logname='run'):
        """Train the model"""

        alpha = 1.0 # learning rate, which can be adjusted later
        n_data = len(X_train)
        n_superbatch = self.n_superbatch

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
                    # collect metrics
                    train_batches += 1
                    train_err += err
                    train_acc += acc

                    if train_batches % 100 == 0:
                        n_total = epoch * n_data + n_batch * train_batches
                        metrics = [n_total, train_err / train_batches, train_acc / train_batches]
                        log_metrics(logname, metrics)

            print "Epoch {} of {} took {:.3f}s ({} minibatches)".format(
                epoch + 1, n_epoch, time.time() - start_time, train_batches)

            # make a full pass over the training data and record metrics:
            train_err, train_acc = evaluate(self.loss, X_train, Y_train, batchsize=1000)
            val_err, val_acc = evaluate(self.loss, X_val, Y_val, batchsize=1000)

            print "  training loss/acc:\t\t{:.6f}\t{:.6f}".format(train_err, train_acc)
            print "  validation loss/acc:\t\t{:.6f}\t{:.6f}".format(val_err, val_acc)

            metrics = [ epoch, train_err, train_acc, val_err, val_acc ]
            log_metrics(logname + '.val', metrics)

    def load_params(self, params):
        """Load a given set of parameters"""
        self.params = params

    def dump_params(self):
        """Dump a given set of parameters"""
        return self.params
