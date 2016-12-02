import pdb
import time
import pickle
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
import lasagne

from model import Model

from layers import GaussianSampleLayer
from distributions import log_bernoulli, log_normal2

# ----------------------------------------------------------------------------

class VAE_REINFORCE(Model):
  """Variational Autoencoder with Gaussian visible and latent variables
     Trained by REINFORCE instead of vanilla SGD"""

  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat    = 200 # latent stochastic variabels
    n_hid    = 500 # size of hidden layer in encoder/decoder
    n_hid_cv = 500 # size of hidden layer in control variate net
    n_out    = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl   = lasagne.nonlinearities.tanh if self.model == 'bernoulli' \
               else T.nnet.softplus

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(
        shape=(None, n_chan, n_dim, n_dim),
        input_var=X)
    l_q_hid = lasagne.layers.DenseLayer(
        l_q_in,
        num_units=n_hid,
        nonlinearity=hid_nl)
    l_q_mu = lasagne.layers.DenseLayer(
        l_q_hid,
        num_units=n_lat,
        nonlinearity=None)
    l_q_logsigma = lasagne.layers.DenseLayer(
        l_q_hid,
        num_units=n_lat,
        nonlinearity=None)
    l_q_sample = GaussianSampleLayer(l_q_mu, l_q_logsigma)

    # create the decoder network
    # pass l_q_sample into l_p_in
    l_p_in = lasagne.layers.InputLayer((None, n_lat))
    l_p_hid = lasagne.layers.DenseLayer(
        l_p_in,
        num_units=n_hid,
        nonlinearity=hid_nl,
        W=lasagne.init.GlorotUniform())
    l_p_mu, l_p_logsigma = None, None

    if self.model == 'bernoulli':
      l_p_mu = lasagne.layers.DenseLayer(
          l_p_hid,
          num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.GlorotUniform(),
          b=lasagne.init.Constant(0.))
      # l_p_sample = l_p_mu

    elif self.model == 'gaussian':
      l_p_mu = lasagne.layers.DenseLayer(
          l_p_hid,
          num_units=n_out,
          nonlinearity=None)
      # relu_shift is for numerical stability - if training data has any
      # dimensions where stdev=0, allowing logsigma to approach -inf
      # will cause the loss function to become NAN. So we set the limit
      # stdev >= exp(-1 * relu_shift)
      relu_shift = 10
      l_p_logsigma = lasagne.layers.DenseLayer(
          l_p_hid,
          num_units=n_out,
          nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift)
      # l_p_sample = GaussianSampleLayer(l_p_mu, l_p_logsigma)

    # create control variate (baseline) network
    l_cv_in = lasagne.layers.InputLayer(
        shape=(None, n_chan, n_dim, n_dim),
        input_var=X)
    l_cv_hid = lasagne.layers.DenseLayer(
        l_cv_in,
        num_units=n_hid_cv,
        nonlinearity=hid_nl)
    l_cv = lasagne.layers.DenseLayer(
        l_cv_hid,
        num_units=1,
        nonlinearity=None)

    # create variables for centering signal
    c = theano.shared(np.zeros((1,1), dtype=np.float64), broadcastable=(True,True))
    v = theano.shared(np.zeros((1,1), dtype=np.float64), broadcastable=(True,True))

    # store certain input layers for downstream (quick hack)
    self.input_layers = {l_q_in, l_p_in, l_cv_in}

    return l_p_mu, l_p_logsigma, \
           l_q_mu, l_q_logsigma, \
           l_q_sample, l_cv, c, v

  def _create_components(self, deterministic=False):
    # load network input
    X = self.inputs[0]
    x = X.flatten(2)

    # load network
    l_p_mu, l_p_logsigma, \
    l_q_mu, l_q_logsigma, \
    l_q_sample, l_cv, c, v = self.network

    # load input layers
    l_q_in, l_p_in, l_cv_in = self.input_layers

    # load network output
    q_mu, q_logsigma, z = lasagne.layers.get_output(
        [l_q_mu, l_q_logsigma, l_q_sample],
        deterministic=deterministic)

    if self.model == 'bernoulli':
      p_mu = lasagne.layers.get_output(
        l_p_mu,
        {l_p_in: z},
        deterministic=deterministic)
    elif self.model == 'gaussian':
      p_mu, p_logsigma = lasagne.layers.get_output(
        [l_p_mu, l_p_logsigma],
        {l_p_in: z},
        deterministic=deterministic)

    # entropy term
    log_qz_given_x = log_normal2(z, q_mu, q_logsigma).sum(axis=1)

    # expected p(x,z) term
    z_prior = T.ones_like(z)*np.float32(0.5)
    log_pz = log_bernoulli(z, z_prior).sum(axis=1)
    if self.model == 'bernoulli':
      log_px_given_z = log_bernoulli(x, p_mu).sum(axis=1)
    elif self.model == 'gaussian':
      log_px_given_z = log_normal2(x, p_mu, p_logsigma).sum(axis=1)
    log_pxz = log_px_given_z + log_pz

    # save them for later
    if deterministic == False:
      self.log_pxz = log_pxz
      self.log_qz_given_x = log_qz_given_x

    return log_pxz.flatten(), log_qz_given_x.flatten()

  def create_objectives(self, deterministic=False):
    # load probabilities
    log_pxz, log_qz_given_x = self._create_components(deterministic=deterministic)

    # compute the lower bound
    elbo = T.mean(log_pxz - log_qz_given_x)

    # we don't use the second accuracy metric right now
    return -elbo, -T.mean(log_qz_given_x)

  def create_gradients(self, loss, deterministic=False):
    from theano.gradient import disconnected_grad as dg

    # load networks
    l_p_mu, l_p_logsigma, \
    l_q_mu, l_q_logsigma, \
    _, l_cv, c, v = self.network

    # load params
    if self.model == 'bernoulli':
      p_params = lasagne.layers.get_all_params([l_p_mu], trainable=True)
    elif self.model == 'gaussian':
      p_params = lasagne.layers.get_all_params([l_p_mu, l_p_logsigma], trainable=True)
    q_params = lasagne.layers.get_all_params([l_q_mu, l_q_logsigma], trainable=True)
    cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)

    # load neural net outputs (probabilities have been precomputed)
    log_pxz, log_qz_given_x = self.log_pxz, self.log_qz_given_x
    cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

    # compute learning signals
    l = log_pxz - log_qz_given_x - cv
    l_avg, l_std = l.mean(), T.maximum(1, l.std())
    c_new = 0.8*c + 0.2*l_avg
    v_new = 0.8*v + 0.2*l_std
    l = (l - c_new) / v_new

    # compute grad wrt p
    p_grads = T.grad(-log_pxz.mean(), p_params)

    elbo = T.mean(log_pxz - log_qz_given_x)

    # compute grad wrt q
    q_target = T.mean(dg(l) * log_qz_given_x)
    # q_grads = T.grad(-0.2*q_target, q_params) # 5x slower rate for q
    q_grads = T.grad(-0.2*elbo, q_params)

    # compute grad of cv net
    cv_target = T.mean(l**2)
    cv_grads = T.grad(cv_target, cv_params)

    # combine and clip gradients
    clip_grad = 1
    max_norm = 5
    grads = p_grads + q_grads + cv_grads
    mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

    return cgrads # concatenated grads

  def get_params(self):
    l_p_mu, l_p_logsigma, \
    l_q_mu, l_q_logsigma, \
    _, l_cv, _, _ = self.network

    if self.model == 'bernoulli':
      p_params = lasagne.layers.get_all_params([l_p_mu], trainable=True)
    elif self.model == 'gaussian':
      p_params = lasagne.layers.get_all_params([l_p_mu, l_p_logsigma], trainable=True)
    q_params = lasagne.layers.get_all_params([l_q_mu, l_q_logsigma], trainable=True)
    cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)

    return p_params + q_params + cv_params

  def create_updates(self, grads, params, alpha, opt_alg, opt_params):
    # call super-class to generate SGD/ADAM updates
    grad_updates = Model.create_updates(self, grads, params, alpha, opt_alg, opt_params)

    # create updates for centering signal
    _, _, _, _, _, l_cv, c, v = self.network
    log_pxz, log_qz_given_x = self.log_pxz, self.log_qz_given_x
    cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

    # compute learning signals
    l = log_pxz - log_qz_given_x - cv
    l_avg, l_std = l.mean(), T.maximum(1, l.std())
    c_new = 0.8*c + 0.2*l_avg
    v_new = 0.8*v + 0.2*l_std

    # compute update for centering signal
    cv_updates = {c : c_new, v : v_new}

    return OrderedDict( grad_updates.items() + cv_updates.items() )
