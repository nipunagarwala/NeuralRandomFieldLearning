import numpy as np
import theano.tensor as T
from lasagne.layers import (
  InputLayer, DenseLayer,
  reshape, flatten,
)
from layers import GumbelSoftmaxSampleLayer
from gsm import GSM


class SBN_GSM(GSM):
  """ Sigmoid Belief Network trained using Gumbel Softmax Reparametrization
      https://arxiv.org/pdf/1611.01144v2.pdf
      https://arxiv.org/pdf/1402.0030.pdf
      (Note: Most of network is identical to GSM)

      With 30 categorical distributions:
      Epoch 200 of 200 took 19.458s (192 minibatches)
        training loss/acc:		    148.459345	-3.388340
        validation loss/acc:		147.884916	-3.387381

      With 100 categorical distributions:
      Epoch 200 of 200 took 31.197s (192 minibatches)
        training loss/acc:		    127.884146	-10.371977
        validation loss/acc:		127.666610	-10.347338

      With 200 categorical distributions:
      Epoch 72 of 200 took 86.836s (192 minibatches)
        training loss/acc:		    119.961985	-16.995254
        validation loss/acc:		119.922936	-16.905882
  """

  def create_model(self, x, y, n_dim, n_out, n_chan=1):
    n_class = 10  # number of classes
    n_cat = 200  # number of categorical distributions
    n_lat = n_class*n_cat  # latent stochastic variables
    n_hid = 500  # size of hidden layer in encoder/decoder
    n_out = n_dim * n_dim * n_chan  # total dimensionality of ouput
    n_in = n_out
    tau = self.tau

    # create the encoder network
    q_net = InputLayer(shape=(None, n_in), input_var=x)
    q_net = DenseLayer(q_net, num_units=n_hid, nonlinearity=T.nnet.relu)
    q_net_mu = DenseLayer(q_net, num_units=n_lat, nonlinearity=T.nnet.sigmoid)
    q_net_mu = reshape(q_net_mu, (-1, n_class))
    # sample from Gumble-Softmax posterior
    q_sample = GumbelSoftmaxSampleLayer(q_net_mu, tau)
    q_sample = reshape(q_sample, (-1, n_cat, n_class))
    # create the decoder network
    p_net = DenseLayer(flatten(q_sample), num_units=n_hid, nonlinearity=T.nnet.relu)
    p_net_mu = DenseLayer(p_net, num_units=n_out, nonlinearity=T.nnet.sigmoid)

    # save network params
    self.n_class = n_class
    self.n_cat = n_cat

    return q_net_mu, p_net_mu
