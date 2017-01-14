
import numpy as np
import theano.tensor as T
from lasagne.layers import (
  InputLayer, DenseLayer, ElemwiseSumLayer, NonlinearityLayer,
  reshape, flatten, get_all_params, get_output,
)
from lasagne.updates import total_norm_constraint
from lasagne.init import GlorotNormal, Normal
from layers import GumbelSoftmaxSampleLayer, GaussianSampleLayer
from distributions import log_bernoulli, log_normal2, log_gumbel_softmax
from gsm import GSM

import theano, lasagne
theano.config.optimizer = 'None'

class ADGM_GSM(GSM):
    """Auxiliary Deep Generative Model trained
    using Gumbel Softmax Reparametrization
    https://arxiv.org/pdf/1602.05473v4.pdf
    https://arxiv.org/pdf/1611.01144v2.pdf

    Epoch 196 of 200 took 71.449s (192 minibatches)
          training loss/acc:		68.463976	-21.743954
          validation loss/acc:		70.561101	-21.519407
    """
    def create_model(self, x, y, n_dim, n_out, n_chan=1):
        n_class = 10  # number of classes
        n_cat = 20  # number of categorical distributions
        n_lat = n_class*n_cat  # latent stochastic variables
        n_aux = 10  # number of auxiliary variables
        n_hid = 500  # size of hidden layer in encoder/decoder
        n_out = n_dim * n_dim * n_chan  # total dimensionality of ouput
        n_in = n_out
        tau = self.tau
        hid_nl = T.nnet.relu
        relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability

        # create the encoder network
        # - create q(a|x)
        qa_net_in = InputLayer(shape=(None, n_in), input_var=x)
        qa_net = DenseLayer(
            qa_net_in, num_units=n_hid,
            W=GlorotNormal('relu'), b=Normal(1e-3),
            nonlinearity=hid_nl,
        )
        qa_net_mu = DenseLayer(
            qa_net, num_units=n_aux,
            W=GlorotNormal(), b=Normal(1e-3),
            nonlinearity=None,
        )
        qa_net_logsigma = DenseLayer(
            qa_net, num_units=n_aux,
            W=GlorotNormal(), b=Normal(1e-3),
            nonlinearity=relu_shift,
        )
        qa_net_sample = GaussianSampleLayer(qa_net_mu, qa_net_logsigma)
        # - create q(z|a, x)
        qz_net_a = DenseLayer(
            qa_net_sample, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        qz_net_b = DenseLayer(
            qa_net_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        qz_net = ElemwiseSumLayer([qz_net_a, qz_net_b])
        qz_net = DenseLayer(
            qz_net, num_units=n_hid,
            nonlinearity=hid_nl
        )
        qz_net_mu = DenseLayer(
            qz_net, num_units=n_lat,
            nonlinearity=None,
        )
        # qz_net_logsigma = DenseLayer(
        #   qz_net, num_units=n_lat,
        #   W=GlorotNormal(),
        #   b=Normal(1e-3),
        #   nonlinearity=relu_shift,
        # )
        # qz_net_sample = GaussianSampleLayer(qz_net_mu, qz_net_logsigma)
        qz_net_mu = reshape(qz_net_mu, (-1, n_class))
        qz_net_sample = GumbelSoftmaxSampleLayer(qz_net_mu, tau)
        qz_net_sample = reshape(qz_net_sample, (-1, n_cat, n_class))

        # create the decoder network
        # - create p(x|z)
        px_net = DenseLayer(
            flatten(qz_net_sample), num_units=n_hid,
            nonlinearity=hid_nl,
        )
        px_net_mu = DenseLayer(
            px_net, num_units=n_out,
            nonlinearity=T.nnet.sigmoid,
        )

        # - create p(a|z)
        pa_net = DenseLayer(
            flatten(qz_net_sample), num_units=n_hid,
            W=GlorotNormal('relu'), b=Normal(1e-3),
            nonlinearity=hid_nl,
        )
        pa_net_mu = DenseLayer(
            pa_net, num_units=n_aux,
            W=GlorotNormal(),
            b=Normal(1e-3),
            nonlinearity=None,
        )
        pa_net_logsigma = DenseLayer(
            pa_net, num_units=n_aux,
            W=GlorotNormal(),
            b=Normal(1e-3),
            nonlinearity=relu_shift,
        )

        # save network params
        self.n_class = n_class
        self.n_cat = n_cat

        return px_net_mu, pa_net_mu, pa_net_logsigma, \
            qz_net_mu, qa_net_mu, qa_net_logsigma, \
            qz_net_sample, qa_net_sample,

    def create_objectives(self, deterministic=False):
        x = self.inputs[0]

        # load network params
        n_class = self.n_class
        n_cat = self.n_cat

        # compute network
        px_mu, pa_mu, pa_logsigma, qz_mu, \
        qa_mu, qa_logsigma, qz_sample, qa_sample = get_output(
            self.network,
            deterministic=deterministic,
        )

        # calculate the likelihoods
        qz_given_ax = T.nnet.softmax(qz_mu)
        log_qz_given_ax = T.log(qz_given_ax + 1e-20)
        entropy = T.reshape(qz_given_ax * (log_qz_given_ax - T.log(1.0 / n_class)), (-1, n_cat, n_class))
        entropy = T.sum(entropy, axis=[1,2])

        log_px_given_z = log_bernoulli(x, px_mu).sum(axis=1)
        log_pa_given_z = log_normal2(qa_sample, pa_mu, pa_logsigma).sum(axis=1)
        log_paxz = log_pa_given_z + log_px_given_z

        # logp(z)+logp(a|z)-logq(a)-logq(z|a)
        elbo = T.mean(log_paxz - entropy)

        return -elbo, -T.mean(entropy)

    def create_gradients(self, loss, deterministic=False):
        grads = GSM.create_gradients(self, loss, deterministic)

        # combine and clip gradients
        clip_grad, max_norm = 1, 5
        mgrads = total_norm_constraint(grads, max_norm=max_norm)
        cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

        return cgrads

    def get_params(self):
        px_net_mu, pa_net_mu = self.network[:2]
        params = get_all_params(px_net_mu, trainable=True)
        params0 = get_all_params(pa_net_mu, trainable=True)

        for param in params0:
            if param not in params:
                params.append(param)

        return params
