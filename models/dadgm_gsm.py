import numpy as np
import theano.tensor as T
from lasagne.layers import (
  InputLayer, DenseLayer, ElemwiseSumLayer, NonlinearityLayer,
  reshape, flatten, get_all_params, get_output, ReshapeLayer
)
from lasagne.updates import total_norm_constraint
from lasagne.init import GlorotNormal, Normal
from layers import GumbelSoftmaxSampleLayer, GaussianSampleLayer, RepeatLayer
from distributions import log_bernoulli, log_normal2, log_gumbel_softmax
from gsm import GSM

import theano, lasagne
theano.config.optimizer = 'None'


class DADGM_GSM(GSM):
    """Discrete Auxiliary Deep Generative Model trained
    using Gumbel Softmax Reparametrization
    https://arxiv.org/pdf/1602.05473v4.pdf
    https://arxiv.org/pdf/1611.01144v2.pdf

    Epoch 200 of 200 took 67.685s (192 minibatches)
        training loss/acc:		56.513651	-21.471892
        validation loss/acc:	59.014807	-21.233923
    """
    def create_model(self, x, y, n_dim, n_out, n_chan=1):
        n_class = 10  # number of classes
        n_cat = 20  # number of categorical distributions
        n_lat = n_class*n_cat  # latent stochastic variables
        n_aux = 10  # number of auxiliary variables
        n_hid = 500  # size of hidden layer in encoder/decoder
        n_sam = self.n_sample = 3  # > 1 b/c we need to sum over a samples for ll
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
        # repeatedly sample
        qa_net_mu = ReshapeLayer(
            RepeatLayer(qa_net_mu, n_ax=1, n_rep=n_sam),
            shape=(-1, n_aux),
        )
        qa_net_logsigma = ReshapeLayer(
            RepeatLayer(qa_net_logsigma, n_ax=1, n_rep=n_sam),
            shape=(-1, n_aux),
        )
        qa_net_sample = GaussianSampleLayer(qa_net_mu, qa_net_logsigma)
        # - create q(z|a, x)
        qz_net_in = lasagne.layers.InputLayer((None, n_aux))
        qz_net_a = DenseLayer(
            qz_net_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        qz_net_b = DenseLayer(
            qa_net_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        qz_net_b = ReshapeLayer(
            RepeatLayer(qz_net_b, n_ax=1, n_rep=n_sam),
            shape=(-1, n_hid),
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

        qz_net_mu = reshape(qz_net_mu, (-1, n_class))
        qz_net_sample = GumbelSoftmaxSampleLayer(qz_net_mu, tau)
        qz_net_sample = reshape(qz_net_sample, (-1, n_cat, n_class))

        # create the decoder network
        # - create p(x|z)
        px_net_in = lasagne.layers.InputLayer((None, n_cat, n_class))
        px_net = DenseLayer(
            flatten(px_net_in), num_units=n_hid,
            nonlinearity=hid_nl,
        )
        px_net_mu = DenseLayer(
            px_net, num_units=n_out,
            nonlinearity=T.nnet.sigmoid,
        )

        # - create p(a|z)
        pa_net = DenseLayer(
            flatten(px_net_in), num_units=n_hid,
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
        self.n_aux = n_aux

        self.input_layers = (qa_net_in, qz_net_in, px_net_in)

        return px_net_mu, pa_net_mu, pa_net_logsigma, \
            qz_net_mu, qa_net_mu, qa_net_logsigma, \
            qz_net_sample, qa_net_sample,

    def create_objectives(self, deterministic=False):
        x = self.inputs[0]

        # duplicate entries to take into account multiple mc samples
        n_sam = self.n_sample
        n_out = x.shape[1]
        x_rep = x.dimshuffle(0,'x',1).repeat(n_sam, axis=1).reshape((-1, n_out))

        # load network params
        n_class = self.n_class
        n_cat = self.n_cat
        n_aux = self.n_aux
        n_lat = n_class*n_cat

        # compute network
        px_net_mu, pa_net_mu, pa_net_logsigma, \
        qz_net_mu, qa_net_mu, qa_net_logsigma, \
        qz_net_sample, qa_net_sample = self.network
        qa_net_in, qz_net_in, px_net_in = self.input_layers

        qa_mu, qa_logsigma, qa_sample = get_output(
            [qa_net_mu, qa_net_logsigma, qa_net_sample],
            deterministic=deterministic,
        )
        qz_mu, qz_sample = get_output(
            [qz_net_mu, qz_net_sample],
            {qz_net_in: qa_sample, qa_net_in: x},
            deterministic=deterministic,
        )
        pa_mu, pa_logsigma = get_output(
            [pa_net_mu, pa_net_logsigma],
            {px_net_in: qz_sample},
            deterministic=deterministic,
        )
        px_mu = get_output(
            px_net_mu,
            {px_net_in: qz_sample},
            deterministic=deterministic,
        )

        # calculate KL(q(z|a)||p(z))
        qz_given_ax = T.nnet.softmax(qz_mu)
        log_qz_given_ax = T.log(qz_given_ax + 1e-20)
        KL = qz_given_ax * (log_qz_given_ax - T.log(1.0 / n_class))
        KL = T.reshape(KL, (-1, n_sam, n_cat, n_class))
        KL = T.sum(KL, axis=[2, 3])
        # calculate \log p(a|z)p(x|z)
        log_px_given_z = log_bernoulli(x_rep, px_mu).sum(axis=1)
        log_pa_given_z = log_normal2(qa_sample, pa_mu, pa_logsigma).sum(axis=1)
        log_paxz = log_pa_given_z + log_px_given_z
        log_paxz = T.reshape(log_paxz, (-1, n_sam))
        # calculate \log q(a)
        log_qa = log_normal2(qa_sample, qa_mu, qa_logsigma).sum(axis=1)
        log_qa = T.reshape(log_qa, (-1, n_sam))

        # calculate elbo
        # \Exp_q(a) [ \log q(a) + \Exp_q(z|a) [ \log p(a|z)p(x|z) ] - KL(q(z|a)||p(z)) ]
        elbo = log_qa + log_paxz - KL

        return -T.mean(elbo), -T.mean(KL)

    def create_gradients(self, loss, deterministic=False):
        grads = GSM.create_gradients(self, loss, deterministic)

        # combine and clip gradients
        clip_grad, max_norm = 1, 5
        mgrads = total_norm_constraint(grads, max_norm=max_norm)
        cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

        return cgrads

    def gen_samples(self, deterministic=False):
        s = self.inputs[-1]
        # pass through decoder
        _, _, px_net_in = self.input_layers
        px_net_mu = self.network[0]

        px_mu = get_output(
            px_net_mu,
            {px_net_in : s},
            deterministic=deterministic,
        )

        return px_mu

    def get_params(self):
        px_net_mu, pa_net_mu, pa_net_logsigma, \
        qz_net_mu, qa_net_mu, qa_net_logsigma, \
        qz_net_sample, qa_net_sample = self.network

        p_params = get_all_params([px_net_mu, pa_net_mu, pa_net_logsigma], trainable=True)
        qa_params = get_all_params(qa_net_sample, trainable=True)
        qz_params = get_all_params(qz_net_sample, trainable=True)

        return p_params + qa_params + qz_params
