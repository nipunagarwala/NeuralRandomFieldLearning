import pdb
import time
import pickle
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import lasagne

from layers import GumbelSoftmaxSampleLayer
from distributions import log_bernoulli
from model import Model


class SBN_GSM(Model):
  """ Sigmoid Belief Network trained using Gumbel Softmax Reparametrization
      https://arxiv.org/pdf/1611.01144v2.pdf
  """
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, opt_alg='adam',
              opt_params={'lr': 1e-3, 'b1': 0.9, 'b2': 0.99}):

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    n_class = 10
    n_cat   = 20  # number of categorical distributions
    n_lat   = n_class*n_cat  # latent stochatic variables
    n_hid   = 500 # size of hidden layer in encoder/decoder
    n_out   = n_dim * n_dim * n_chan
    hid_nl  = lasagne.nonlinearities.tanh

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(
      shape=(None, n_chan, n_dim, n_dim), input_var=X)
    l_q_hid = lasagne.layers.DenseLayer(
      l_q_in, num_units=n_hid, nonlinearity=hid_nl)
    l_q_mu = lasagne.layers.DenseLayer(
      l_q_hid, num_units=n_lat, nonlinearity=T.nnet.sigmoid)
    l_q_mu = lasagne.layers.ReshapeLayer(l_q_mu, (-1, n_class))

    # sample from Gumble-Softmax posterior
    tau = theano.shared(1.0, name="temperature")
    l_q_sample = GumbelSoftmaxSampleLayer(
      l_q_mu, temperature=tau, hard=False)
    l_q_sample = lasagne.layers.ReshapeLayer(l_q_sample, (-1, n_cat, n_class))

    # create the decoder network
    l_p_in = lasagne.layers.InputLayer((None, n_lat))
    l_p_hid = lasagne.layers.DenseLayer(
      l_p_in, num_units=n_hid, nonlinearity=hid_nl,
      W=lasagne.init.GlorotUniform())
    l_p_mu = lasagne.layers.DenseLayer(
      l_p_hid, num_units=n_out,
      nonlinearity = lasagne.nonlinearities.sigmoid,
      W=lasagne.init.GlorotUniform(),
      b=lasagne.init.Constant(0.))

    # save network params
    self.n_class = n_class
    self.n_cat   = n_cat

    return l_p_mu, l_q_mu, l_q_sample

  def create_objectives(self, deterministic=False):
    X = self.inputs[0]
    x = X.flatten(2)

    # load network params
    n_class = self.n_class
    n_cat   = self.n_cat

    # load network output
    l_p_mu, l_q_mu, l_q_sample = self.network
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
    l_p_mu, l_q_mu, _ = self.network
    p_params = lasagne.layers.get_all_params(l_p_mu, trainable=True)
    q_params = lasagne.layers.get_all_params(l_q_mu, trainable=True)
    return p_params + q_params
