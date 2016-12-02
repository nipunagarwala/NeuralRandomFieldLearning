import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

from layers import GaussianSampleLayer
from distributions import log_bernoulli, log_normal, log_normal2

# ----------------------------------------------------------------------------

class VAE_REINFORCE(Model):
  """Variational Autoencoder with Gaussian visible and latent variables"""
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat = 200 # latent stochastic variabels
    n_hid = 500 # size of hidden layer in encoder/decoder
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.tanh if self.model == 'bernoulli' \
             else T.nnet.softplus

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim),
                                     input_var=X)
    l_q_hid = lasagne.layers.DenseLayer(
        l_q_in, num_units=n_hid,
        nonlinearity=hid_nl)
    l_q_mu = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=None)
    l_q_logsigma = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=None)

    # create the decoder network
    l_q = GaussianSampleLayer(l_q_mu, l_q_logsigma)

    l_p_hid = lasagne.layers.DenseLayer(
        l_q, num_units=n_hid,
        nonlinearity=hid_nl,
        W=lasagne.init.GlorotUniform())
    l_p_mu, l_p_logsigma = None, None

    if self.model == 'bernoulli':
      l_p_mu = lasagne.layers.DenseLayer(l_p_hid, num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.GlorotUniform(),
          b=lasagne.init.Constant(0.))

    elif self.model == 'gaussian':
      l_p_mu = lasagne.layers.DenseLayer(
          l_p_hid, num_units=n_out,
          nonlinearity=None)
      # relu_shift is for numerical stability - if training data has any
      # dimensions where stdev=0, allowing logsigma to approach -inf
      # will cause the loss function to become NAN. So we set the limit
      # stdev >= exp(-1 * relu_shift)
      relu_shift = 10
      l_p_logsigma = lasagne.layers.DenseLayer(
          l_p_hid, num_units=n_out,
          nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift)

      l_sample = GaussianSampleLayer(l_p_mu, l_p_logsigma)

    return l_p_mu, l_p_logsigma, l_q_mu, l_q_logsigma, l_q

  def create_objectives(self, deterministic=False):
    # load network input
    X = self.inputs[0]
    x = X.flatten(2)

    l_p_mu, l_p_logsigma, l_q_mu, l_q_logsigma, l_q = self.network

    q_mu, q_logsigma, z = lasagne.layers.get_output(
      [l_q_mu, l_q_logsigma, l_q],
      deterministic=deterministic)

    # load network output
    if self.model == 'bernoulli':
      p_mu = lasagne.layers.get_output(l_p_mu, deterministic=deterministic)
    elif self.model == 'gaussian':
      p_mu, p_logsigma = lasagne.layers.get_output(
        [l_p_mu, l_p_logsigma],
        deterministic=deterministic)

    log_qz_given_x = log_normal2(z, q_mu, q_logsigma).sum(axis=1)

    z_prior_sigma = T.cast(T.ones_like(q_logsigma), dtype=theano.config.floatX)
    z_prior_mu = T.cast(T.zeros_like(q_mu), dtype=theano.config.floatX)
    log_pz = log_normal(z, z_prior_mu,  z_prior_sigma).sum(axis=1)

    if self.model == 'bernoulli':
      log_px_given_z = log_bernoulli(x, p_mu).sum(axis=1)
    elif self.model == 'gaussian':
      log_px_given_z = log_normal2(x, p_mu, p_logsigma).sum(axis=1)

    log_pxz = log_px_given_z + log_pz

    # compute the evidence lower bound
    elbo = T.mean(log_pxz - log_qz_given_x)

    # we don't use the spearate accuracy metric right now
    return -elbo, -T.mean(log_qz_given_x)

  def create_gradients(self, loss, deterministic=False):
    grads = Model.create_gradients(self, loss, deterministic)

    # combine and clip gradients
    clip_grad = 1
    max_norm = 5
    mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

    return cgrads

  def get_params(self):
    l_p_mu, l_p_logsigma, l_q_mu, l_q_logsigma, _ = self.network
    if self.model == 'bernoulli':
      p_params = lasagne.layers.get_all_params([l_p_mu], trainable=True)
    elif self.model == 'gaussian':
      p_params = lasagne.layers.get_all_params([l_p_mu, l_p_logsigma], trainable=True)
    q_params = lasagne.layers.get_all_params([l_q_mu, l_q_logsigma], trainable=True)
    return p_params + q_params
