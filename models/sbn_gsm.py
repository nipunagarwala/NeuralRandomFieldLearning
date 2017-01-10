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

      Epoch 200 of 200 took 22.430s (192 minibatches)
        training loss/acc:		  96.704909	-22.189577
        validation loss/acc:    98.460990	-21.986126
  """

  def create_model(self, x, y, n_dim, n_out, n_chan=1):
    n_class = 10  # number of classes
    n_cat = 30  # number of categorical distributions
    n_lat = n_class*n_cat  # latent stochastic variables
    n_hid = 500  # size of hidden layer in encoder/decoder
    n_out = n_dim * n_dim * n_chan  # total dimensionality of ouput
    n_in = n_out
    tau = self.tau

    # create the encoder network
    q_net = InputLayer(shape=(None, n_in), input_var=x)
    q_net = DenseLayer(q_net, num_units=n_hid, nonlinearity=T.nnet.relu)
    q_net_mu = DenseLayer(q_net, num_units=n_lat, nonlinearity=None)
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
