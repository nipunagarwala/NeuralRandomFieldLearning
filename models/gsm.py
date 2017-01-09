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
  """
  Gumbel Softmax w/ categorical latent variables
  https://arxiv.org/pdf/1611.01144v2.pdf

  Epoch 100 of 100 took 60.530s (500 minibatches)
    training loss/acc:		  95.905443	-19.943117
    validation loss/acc:		98.537678	-19.804097
  """
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800,
              opt_alg='adam', opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}):
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

    # create shared learning variables
    tau = theano.shared(
        np.float32(5.0), name='temperature',
        allow_downcast=True, borrow=False,
    )
    self.tau = tau

    # create input vars
    x = T.matrix(dtype=theano.config.floatX)
    y = T.ivector()
    idx1, idx2 = T.lscalar(), T.lscalar()
    self.inputs = (x, y, idx1, idx2)

    # create lasagne model
    self.network = self.create_model(x, y, n_dim, n_out, n_chan)

    # create objectives
    loss, acc = self.create_objectives(deterministic=False)
    self.objectives = (loss, acc)

    # create gradients
    grads = self.create_gradients(loss, deterministic=False)

    # get params
    params = self.get_params()

    # create updates
    alpha = T.scalar(dtype=theano.config.floatX)  # adjustable learning rate
    updates = self.create_updates(grads, params, alpha, opt_alg, opt_params)

    self.train = theano.function(
      [idx1, idx2, alpha], [loss, acc],
      updates=updates,
      givens={
        x : train_set_x[idx1:idx2],
        y : train_set_y_int[idx1:idx2]
      },
      on_unused_input='warn',
    )

    self.loss = theano.function([x, y], [loss, acc], on_unused_input='warn')

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
    self.params = params
    self.grads = (grads, None)
    self.metrics = (loss, acc)

  def create_model(self, x, y, n_dim, n_out, n_chan=1):
    n_class = 10  # number of classes
    n_cat   = 30  # number of categorical distributions
    n_out   = n_dim * n_dim * n_chan
    n_in    = n_out
    tau     = self.tau

    # create the encoder network
    net = InputLayer(shape=(None, n_in), input_var=x)
    net = DenseLayer(net, num_units=512, nonlinearity=T.nnet.relu)
    net = DenseLayer(net, num_units=256, nonlinearity=T.nnet.relu)
    # sample from Gumble-Softmax posterior
    logits_y = DenseLayer(net, n_cat*n_class, nonlinearity=None)
    logits_y = reshape(logits_y, (-1, n_class))
    y = GumbelSoftmaxSampleLayer(logits_y, tau)
    y = reshape(y, (-1, n_cat, n_class))
    # create the decoder network
    net = DenseLayer(flatten(y), 256, nonlinearity=T.nnet.relu)
    net = DenseLayer(net, 512, nonlinearity=T.nnet.relu)
    logits_x = DenseLayer(net, n_out, nonlinearity=T.nnet.sigmoid)

    # save network params
    self.n_class = n_class
    self.n_cat = n_cat

    return logits_y, logits_x

  def create_objectives(self, deterministic=False):
    x = self.inputs[0]

    # load network params
    n_class = self.n_class
    n_cat = self.n_cat

    # load network output
    logits_y, logits_x = self.network
    _logits_y, _logits_x = lasagne.layers.get_output([logits_y, logits_x])

    # define the loss
    q_y = T.nnet.softmax(_logits_y)
    log_q_y = T.log(q_y + 1e-20)
    log_p_x = log_bernoulli(x, _logits_x)

    kl_tmp = T.reshape(q_y * (log_q_y - T.log(1.0 / n_class)), [-1 , n_cat, n_class])
    KL = T.sum(kl_tmp, axis=[1, 2])
    elbo = T.sum(log_p_x, axis=1) - KL
    loss = T.mean(-elbo)

    return loss, -T.mean(KL)

  def get_params(self):
    _, logits_x = self.network
    return get_all_params(logits_x)

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
