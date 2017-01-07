import pdb
import time
import pickle
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import lasagne

from layers import GumbelSoftmaxSampleLayer
from distributions import log_bernoulli
from model import Model
from helpers import *


def evaluate(eval_f, X, Y, tau, batchsize=1000):
  tot_err, tot_acc, batches = 0, 0, 0
  for inputs, targets in iterate_minibatches(X, Y, batchsize, shuffle=False):
    err, acc = eval_f(inputs, targets, tau)
    tot_err += err
    tot_acc += acc
    batches += 1
  return tot_err / batches, tot_acc / batches


class GSM(Model):
  """ Gumbel Softmax w/ categorical latent variables
      https://arxiv.org/pdf/1611.01144v2.pdf
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
      ), borrow=False
    )
    val_set_x = theano.shared(
      np.empty(
        (n_superbatch, n_chan*n_dim*n_dim),
        dtype=theano.config.floatX
      ), borrow=False
    )

    # create y-variables
    train_set_y = theano.shared(
      np.empty(
        (n_superbatch,),
        dtype=theano.config.floatX
      ), borrow=False
    )
    val_set_y = theano.shared(
      np.empty(
        (n_superbatch,),
        dtype=theano.config.floatX
      ), borrow=False
    )
    train_set_y_int = T.cast(train_set_y, 'int32')
    val_set_y_int = T.cast(val_set_y, 'int32')

    # create input vars
    X = T.matrix(dtype=theano.config.floatX)
    Y = T.ivector()
    idx1, idx2 = T.lscalar(), T.lscalar()
    self.inputs = (X, Y, idx1, idx2)

    # create lasagne model
    self.network = self.create_model(X, Y, n_dim, n_out, n_chan)

    # create objectives
    loss, acc            = self.create_objectives(deterministic=False)
    loss_test, acc_test  = self.create_objectives(deterministic=True)
    self.objectives      = (loss, acc)
    self.objectives_test = (loss_test, acc_test)

    # load params
    params = self.get_params()

    # create gradients
    grads      = self.create_gradients(loss, deterministic=False)
    grads_test = self.create_gradients(loss_test, deterministic=True)

    # create updates
    alpha = T.scalar(dtype=theano.config.floatX) # adjustable learning rate
    tau = T.scalar(dtype=theano.config.floatX) # adjustable tau
    updates = self.create_updates(
      grads, params, alpha, tau,
      opt_alg, opt_params
    )

    # create methods for training / prediction
    self.train = theano.function(
      [idx1, idx2, alpha, tau], [loss, acc],
      updates=updates,
      givens={
        X : train_set_x[idx1:idx2],
        Y : train_set_y_int[idx1:idx2]
      },
      on_unused_input='warn'
    )
    self.loss = theano.function(
      [X, Y, tau], [loss, acc],
      on_unused_input='warn'
    )

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
    self.grads = (grads, grads_test)
    self.metrics = (loss, acc)

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    hid_nl = lasagne.nonlinearities.rectify
    n_class = 10
    n_cat = 30  # number of categorical distributions
    n_out = n_dim * n_dim * n_chan
    n_in = n_out
    tau = theano.shared(
      1.0, name="temperature",
      allow_downcast=True,
    )

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(
      shape=(None, n_in), input_var=X)
    l_q_hid1 = lasagne.layers.DenseLayer(
      l_q_in, num_units=512, nonlinearity=hid_nl)
    l_q_hid2 = lasagne.layers.DenseLayer(
      l_q_hid1, num_units=256, nonlinearity=hid_nl)

    # create the bottleneck
    l_q_mu = lasagne.layers.DenseLayer(
      l_q_hid2, num_units=n_class*n_cat, nonlinearity=None)
    l_q_mu = lasagne.layers.reshape(l_q_mu, (-1, n_class))
    # sample from Gumble-Softmax posterior
    l_q_sample = GumbelSoftmaxSampleLayer(l_q_mu, tau)
    l_q_sample = lasagne.layers.ReshapeLayer(l_q_sample, (-1, n_cat, n_class))

    # create the decoder network
    l_p_in = lasagne.layers.InputLayer((None, n_cat, n_class))
    l_p_in = lasagne.layers.flatten(l_p_in)
    l_p_hid1 = lasagne.layers.DenseLayer(
      l_p_in, num_units=256, nonlinearity=hid_nl)
    l_p_hid2 = lasagne.layers.DenseLayer(
      l_p_hid1, num_units=512, nonlinearity=hid_nl)
    l_p_mu = lasagne.layers.DenseLayer(
      l_p_hid2, num_units=n_out, nonlinearity=None)

    # save network params
    self.n_class = n_class
    self.n_cat   = n_cat

    return l_p_mu, l_q_mu, l_q_sample, tau

  def create_objectives(self, deterministic=False):
    X = self.inputs[0]
    x = X.flatten(2)

    # load network params
    n_class = self.n_class
    n_cat   = self.n_cat

    # load network output
    l_p_mu, l_q_mu, l_q_sample, _ = self.network
    z, q_mu = lasagne.layers.get_output([l_q_sample, l_q_mu], deterministic=deterministic)
    p_mu = lasagne.layers.get_output(l_p_mu, z, deterministic=deterministic)

    q_z = T.nnet.softmax(q_mu)
    log_q_z = T.log(q_z + 1e-20)
    log_p_x = log_bernoulli(x, p_mu)

    kl_tmp = T.reshape(q_z * (log_q_z - T.log(1.0 / n_class)), [-1 , n_cat, n_class])
    kl = T.sum(kl_tmp, axis=[1, 2])
    elbo = T.mean(T.sum(log_p_x, 1) - kl)

    return -elbo, -T.mean(kl)

  def get_params(self):
    l_p_mu, l_q_mu, _, _ = self.network
    p_params = lasagne.layers.get_all_params(l_p_mu, trainable=True)
    q_params = lasagne.layers.get_all_params(l_q_mu, trainable=True)
    return p_params + q_params

  def create_updates(self, grads, params, alpha, tau, opt_alg, opt_params):
    scaled_grads = [grad * alpha for grad in grads]
    lr = opt_params.get('lr', 1e-3)
    if opt_alg == 'sgd':
      grad_updates = lasagne.updates.sgd(
        scaled_grads, params,
        learning_rate=lr,
      )
    elif opt_alg == 'adam':
      b1, b2 = opt_params.get('b1', 0.9), opt_params.get('b2', 0.999)
      grad_updates = lasagne.updates.adam(
        scaled_grads, params,
        learning_rate=lr, beta1=b1, beta2=b2,
      )
    else:
      grad_updates = OrderedDict()

    # get tau updates
    _, _, _, nn_tau = self.network
    tau_updates = { nn_tau : tau }

    return OrderedDict( grad_updates.items() + tau_updates.items() )

  def fit(
    self, X_train, Y_train, X_val, Y_val,
    n_epoch=10, n_batch=100, logname='run'
  ):
    """Train the model"""

    tau0 = 1.0  # initial temp
    min_tau = 0.5  # minimum temp
    tau_anneal_rate = 0.00003  # adjusting rate
    alpha = 1.0  # learning rate, which can be adjusted later
    n_data = len(X_train)
    n_superbatch = self.n_superbatch
    n_exec = 0  # track # of train() executions

    # flatten X_train, X_test
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
          if n_exec % 1000 == 0:
            cur_tau = np.maximum(tau0*np.exp(-tau_anneal_rate*n_exec), min_tau)
            print "- Annealing tau to {} ({} execs)".format(cur_tau, n_exec)
            if n_exec > 0:
              alpha *= 0.9
              print "- Annealing alpha to {} ({} execs)".format(alpha, n_exec)
          err, acc = self.train(idx1, idx2, alpha, cur_tau)

          # collect metrics
          n_exec += 1
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
        train_batches
      )

      # make a full pass over the training data and record metrics:
      train_err, train_acc = evaluate(
        self.loss, X_train, Y_train,
        cur_tau, batchsize=1000,
      )
      val_err, val_acc = evaluate(
        self.loss, X_val, Y_val,
        cur_tau, batchsize=1000,
      )

      print "  training loss/acc:\t\t{:.6f}\t{:.6f}".format(train_err, train_acc)
      print "  validation loss/acc:\t\t{:.6f}\t{:.6f}".format(val_err, val_acc)

      metrics = [ epoch, train_err, train_acc, val_err, val_acc ]
      log_metrics(logname + '.val', metrics)
