import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

from layers import GaussianSampleLayer
from distributions import log_bernoulli, log_normal2

# ----------------------------------------------------------------------------

class VAE(Model):
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
    n_hid_cv = 500 # size of hidden layer in control variate net
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.tanh if self.model == 'bernoulli' \
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
      # sample is bernoulli
      l_p_sample = l_p_mu

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
      l_p_sample = GaussianSampleLayer(l_p_mu, l_p_logsigma)

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

    return l_p_mu, l_p_logsigma, \
           l_q_mu, l_q_logsigma, \
           l_q_sample, l_p_sample, \
           l_cv, c, v

  def create_objectives(self, deterministic=False):
    # load network input
    X = self.inputs[0]

    # load network
    l_p_mu, l_p_logsigma, \
    l_q_mu, l_q_logsigma, \
    l_q_sample, l_p_sample, \
    l_cv, c, v = self.network

    # load network output
    if self.model == 'bernoulli':
      q_mu, q_logsigma, q_sample \
          = lasagne.layers.get_output(
              [l_q_mu, l_q_logsigma, l_q_sample],
              deterministic=deterministic)
      p_sample = lasagne.layers.get_output(
          l_p_sample,
          {l_p_in: q_sample},
          deterministic=deterministic)
    elif self.model == 'gaussian':
      q_mu, q_logsigma, q_sample \
          = lasagne.layers.get_output(
              [l_q_mu, l_q_logsigma, l_q_sample],
              deterministic=deterministic)
      p_mu, p_logsigma, p_sample \
          = lasagne.layers.get_output(
              [l_p_mu, l_p_logsigma, l_p_sample],
              {l_p_in: q_sample},
              deterministic=deterministic)

    # first term of the ELBO: kl-divergence (using the closed form expression)
    kl_div = 0.5 * T.sum(1 + 2*q_logsigma - T.sqr(q_mu)
                         - T.exp(2 * q_logsigma), axis=1).mean()
    log_qz_given_x = log_normal2(l_p_z, l_q_mu, l_q_logsigma).sum(axis=1)

    # second term: log-likelihood of the data under the model
    if self.model == 'bernoulli':
      log_pxz = -lasagne.objectives.binary_crossentropy(p_sample, X.flatten(2)).sum(axis=1)
    elif self.model == 'gaussian':
      def log_lik(x, mu, log_sig):
          return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + log_sig)
                        - 0.5 * T.sqr(x - mu) / T.exp(2 * log_sig), axis=1)

    loss = -1 * (log_pxz.mean() + kl_div)

    self.log_pxz = log_pxz
    self.log_qz_given_x = log_qz_given_x

    # we don't use the spearate accuracy metric right now
    return loss, -kl_div

  def create_gradients(self, loss, deterministic=False):
    from theano.gradient import disconnected_grad as dg

    # load networks
    l_p_mu, l_p_logsigma, \
    l_q_mu, l_q_logsigma, \
    l_q_sample, l_p_sample, \
    l_cv, c, v = self.network

    # load params
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

    # compute grad wrt q
    q_target = T.mean(dg(l) * log_qz_given_x)
    q_grads = T.grad(-0.2*q_target, q_params) # 5x slower rate for q

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
    _, _, _, _, _, l_p_sample, l_cv, c, v = self.network
    pq_params = lasagne.layers.get_all_params(l_p_sample, trainable=True)
    cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)
    return pq_params + cv_params

  def create_updates(self, grads, params, alpha, opt_alg, opt_params):
    # call super-class to generate SGD/ADAM updates
    grad_updates = Model.create_updates(self, grads, params, alpha, opt_alg, opt_params)

    # create updates for centering signal
    _, _, _, _, _, _, l_cv, c, v = self.network
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
