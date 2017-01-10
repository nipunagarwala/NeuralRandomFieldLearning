import theano.tensor as T
from lasagne.layers import (
  InputLayer, DenseLayer, ElemwiseSumLayer,
  reshape, flatten, get_all_params, get_output,
)
from lasagne.init import GlorotNormal, Normal
from layers import GumbelSoftmaxSampleLayer, GaussianSampleLayer
from distributions import log_bernoulli, log_normal2
from gsm import GSM

class DADGM_GSM(GSM):
  """ Discrete Auxiliary Deep Generative Model trained
      using Gumbel Softmax Reparametrization

      https://arxiv.org/pdf/1602.05473v4.pdf
      https://arxiv.org/pdf/1611.01144v2.pdf
  """
  def create_model(self, x, y, n_dim, n_out, n_chan=1):
    n_class = 10  # number of classes
    n_cat = 30  # number of categorical distributions
    n_lat = n_class*n_cat  # latent stochastic variables
    n_aux = 10  # number of auxiliary variables
    n_hid = 500  # size of hidden layer in encoder/decoder
    n_out = n_dim * n_dim * n_chan  # total dimensionality of ouput
    n_in = n_out
    tau = self.tau

    # create the encoder network
    # - create q(a|x)
    qa_net_in = InputLayer(shape=(None, n_in), input_var=x)
    qa_net_mu = DenseLayer(
      qa_net_in, num_units=n_aux,
      nonlinearity=None,
    )
    qa_net_logsigma = DenseLayer(
      qa_net_in, num_units=n_aux,
      nonlinearity=T.nnet.relu,
    )
    qa_net_sample = GaussianSampleLayer(qa_net_mu, qa_net_logsigma)
    # - create q(z|a, x)
    qz_net_in = InputLayer((None, n_aux))
    qz_net_a = DenseLayer(
      qz_net_in, num_units=n_hid,
      nonlinearity=T.nnet.relu,
    )
    qz_net_b = DenseLayer(
      qa_net_in, num_units=n_hid,
      nonlinearity=T.nnet.relu,
    )
    qz_net = ElemwiseSumLayer([qz_net_a, qz_net_b])
    qz_net = DenseLayer(
      qz_net, num_units=n_hid,
      nonlinearity=T.nnet.relu,
    )
    qz_net_mu = DenseLayer(
      qz_net, num_units=n_lat,
      nonlinearity=None,
    )
    qz_net_mu = reshape(qz_net_mu, (-1, n_class))
    # - sample from Gumble-Softmax posterior
    qz_net_sample = GumbelSoftmaxSampleLayer(qz_net_mu, tau)
    qz_net_sample = reshape(qz_net_sample, (-1, n_cat, n_class))
    qz_net_sample = flatten(qz_net_sample)
    # create the decoder network
    # - create p(x|z)
    px_net_in = InputLayer((None, n_lat))
    px_net = DenseLayer(
      px_net_in, num_units=n_hid,
      nonlinearity=T.nnet.relu,
    )
    px_net_mu = DenseLayer(
      px_net, num_units=n_out,
      nonlinearity=T.nnet.sigmoid,
    )
    # - create p(a|z)
    pa_net = DenseLayer(
      px_net_in, num_units=n_hid,
      nonlinearity=T.nnet.relu,
    )
    pa_net_mu = DenseLayer(
      pa_net, num_units=n_aux,
      nonlinearity=None,
    )
    pa_net_logsigma = DenseLayer(
      pa_net, num_units=n_aux,
      nonlinearity=T.nnet.relu,
      W=GlorotNormal(), b=Normal(1e-3),
    )

    # save network params
    self.n_class = n_class
    self.n_cat = n_cat

    # store input layers
    self.input_layers = (qa_net_in, qz_net_in, px_net_in)

    return px_net_mu, pa_net_mu, pa_net_logsigma, \
      qa_net_mu, qa_net_logsigma, qz_net_mu, \
      qa_net_sample, qz_net_sample

  def create_objectives(self, deterministic=False):
    x = self.inputs[0]

    # load network params
    n_class = self.n_class
    n_cat = self.n_cat

    # load network output
    px_net_mu, pa_net_mu, pa_net_logsigma, \
    qa_net_mu, qa_net_logsigma, qz_net_mu, \
    qa_net_sample, qz_net_sample = self.network

    qa_net_in, qz_net_in, px_net_in = self.input_layers
    qa_mu, qa_logsigma, qa_sample = get_output(
      [qa_net_mu, qa_net_logsigma, qa_net_sample],
      deterministic=deterministic,
    )
    qz_mu, qz_sample = get_output(
      [qz_net_mu, qz_net_sample],
      {qz_net_in : qa_sample, qa_net_in : x},
      deterministic=deterministic,
    )
    pa_mu, pa_logsigma = get_output(
      [pa_net_mu, pa_net_logsigma],
      {px_net_in : qz_sample},
      deterministic=deterministic,
    )
    px_mu = get_output(
      px_net_mu, {px_net_in : qz_sample},
      deterministic=deterministic,
    )

    # calculate the likelihoods
    log_px = log_bernoulli(x, px_mu)
    log_pa = log_normal2(qa_sample, pa_mu, pa_logsigma)
    log_pxa = log_px + log_pa

    qz = T.nnet.softmax(qz_mu)
    log_qz = T.log(qz + 1e-20)
    log_qa = log_normal2(qa_sample, qa_mu, qa_logsigma)
    log_qza = log_qz + log_qa

    # calculate KL divergence
    kl_tmp = T.reshape(
      qz * (log_qza - T.log(1.0 / n_class)),
      [-1 , n_cat, n_class],
    )
    KL = T.sum(kl_tmp, axis=[1, 2])
    elbo = T.sum(log_pxa, axis=1) - KL
    loss = T.mean(-elbo)

    return loss, -T.mean(KL)

  def get_params(self):
    # load network
    px_net_mu, pa_net_mu, pa_net_logsigma, \
    qa_net_mu, _, _, _, qz_net_sample = self.network

    # load params individually
    p_params = get_all_params(
      [px_net_mu, pa_net_mu, pa_net_logsigma],
      trainable=True,
    )
    qa_params = get_all_params(qa_net_mu, trainable=True)
    qz_params = get_all_params(qz_net_sample, trainable=True)

    return p_params + qa_params + qz_params
