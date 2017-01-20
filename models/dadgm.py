import time
import pickle
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
import lasagne
from theano.gradient import disconnected_grad as dg

from model import Model
from layers.shape import RepeatLayer
from layers import (
    GaussianSampleLayer,
    BernoulliSampleLayer,
    GaussianMultiSampleLayer,
)
from distributions import log_bernoulli, log_normal2


class DADGM(Model):
    """Auxiliary Deep Generative Model (unsupervised version) with discrete z"""
    def __init__(
        self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='bernoulli',
        opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}
    ):
        # save model that wil be created
        self.model = model
        self.n_sample = 1 # adjustable parameter, though 1 works best in practice
        Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

    def create_model(self, X, Y, n_dim, n_out, n_chan=1):
        # params
        n_lat = 200  # latent stochastic variables
        n_aux = 10  # auxiliary variables
        n_hid = 500  # size of hidden layer in encoder/decoder
        n_sam = self.n_sample  # number of monte-carlo samples
        n_hid_cv = 500  # size of hidden layer in control variate net
        n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
        hid_nl = lasagne.nonlinearities.tanh
        relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability

        # create the encoder network
        # create q(a|x)
        l_qa_in = lasagne.layers.InputLayer(
            shape=(None, n_chan, n_dim, n_dim),
            input_var=X,
        )
        l_qa_hid = lasagne.layers.DenseLayer(
            l_qa_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        l_qa_mu = lasagne.layers.DenseLayer(
            l_qa_in, num_units=n_aux,
            nonlinearity=None,
        )
        l_qa_logsigma = lasagne.layers.DenseLayer(
            l_qa_in, num_units=n_aux,
            nonlinearity=relu_shift,
        )
        # repeatedly sample
        l_qa_mu = lasagne.layers.ReshapeLayer(
            RepeatLayer(l_qa_mu, n_ax=1, n_rep=n_sam),
            shape=(-1, n_aux),
        )
        l_qa_logsigma = lasagne.layers.ReshapeLayer(
            RepeatLayer(l_qa_logsigma, n_ax=1, n_rep=n_sam),
            shape=(-1, n_aux),
        )
        l_qa = GaussianSampleLayer(l_qa_mu, l_qa_logsigma)

        # create q(z|a,x)
        l_qz_in = lasagne.layers.InputLayer((None, n_aux))
        l_qz_hid1a = lasagne.layers.DenseLayer(
            l_qz_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        l_qz_hid1b = lasagne.layers.DenseLayer(
            l_qa_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        l_qz_hid1b = lasagne.layers.ReshapeLayer(
            RepeatLayer(l_qz_hid1b, n_ax=1, n_rep=n_sam),
            shape=(-1, n_hid),
        )
        l_qz_hid2 = lasagne.layers.ElemwiseSumLayer([l_qz_hid1a, l_qz_hid1b])
        # l_qz_hid2 = lasagne.layers.ConcatLayer([l_qz_hid1a, l_qz_hid1b])
        # l_qz_hid2 = lasagne.layers.NonlinearityLayer(l_qz_hid2, hid_nl)
        # test w/o a:
        l_qz_hid3 = lasagne.layers.DenseLayer(
            l_qz_hid2, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        l_qz_mu = lasagne.layers.DenseLayer(
            l_qz_hid3, num_units=n_lat,
            nonlinearity=T.nnet.sigmoid,
        )
        l_qz = BernoulliSampleLayer(l_qz_mu)
        l_qz_logsigma = None

        # create the decoder network
        # create p(x|z)
        l_px_in = lasagne.layers.InputLayer((None, n_lat))
        l_px_hid = lasagne.layers.DenseLayer(
            l_px_in, num_units=n_hid,
            W=lasagne.init.GlorotUniform(),
            nonlinearity=hid_nl,
        )
        l_px_mu, l_px_logsigma = None, None

        if self.model == 'bernoulli':
            l_px_mu = lasagne.layers.DenseLayer(
                l_px_hid, num_units=n_out,
                nonlinearity=lasagne.nonlinearities.sigmoid,
            )
        elif self.model == 'gaussian':
            l_px_mu = lasagne.layers.DenseLayer(
                l_px_hid, num_units=n_out,
                nonlinearity=None,
            )
            l_px_logsigma = lasagne.layers.DenseLayer(
                l_px_hid, num_units=n_out,
                nonlinearity=relu_shift,
            )

        # create p(a|z)
        l_pa_hid = lasagne.layers.DenseLayer(
            l_px_in, num_units=n_hid,
            nonlinearity=hid_nl,
        )
        l_pa_mu = lasagne.layers.DenseLayer(
            l_pa_hid, num_units=n_aux,
            nonlinearity=None,
        )
        l_pa_logsigma = lasagne.layers.DenseLayer(
            l_pa_hid, num_units=n_aux,
            W=lasagne.init.GlorotNormal(),
            b=lasagne.init.Normal(1e-3),
            nonlinearity=relu_shift,
        )

        # create control variate (baseline) network
        l_cv_in = lasagne.layers.InputLayer(
            shape=(None, n_chan, n_dim, n_dim),
            input_var=X,
        )
        l_cv_hid = lasagne.layers.DenseLayer(
            l_cv_in, num_units=n_hid_cv,
            nonlinearity=hid_nl,
        )
        l_cv = lasagne.layers.DenseLayer(
            l_cv_hid, num_units=1,
            nonlinearity=None,
        )

        # create variables for centering signal
        c = theano.shared(np.zeros((1,1), dtype=np.float64), broadcastable=(True,True))
        v = theano.shared(np.zeros((1,1), dtype=np.float64), broadcastable=(True,True))

        # store certain input layers for downstream (quick hack)
        self.input_layers = (l_qa_in, l_qz_in, l_px_in, l_cv_in)
        self.n_lat = n_lat
        self.n_hid = n_hid

        return l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
               l_qa_mu, l_qa_logsigma, l_qz_mu, l_qz_logsigma, \
               l_qa, l_qz, l_cv, c, v

    def _create_components(self, deterministic=False):
        # load network input
        X = self.inputs[0]
        x = X.flatten(2)

        # duplicate entries to take into account multiple mc samples
        n_sam = self.n_sample
        n_out = x.shape[1]
        x = x.dimshuffle(0,'x',1).repeat(n_sam, axis=1).reshape((-1, n_out))

        # load networks
        l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
        l_qa_mu, l_qa_logsigma, l_qz_mu, l_qz_logsigma, l_qa, l_qz, _, _, _ = self.network
        l_qa_in, l_qz_in, l_px_in, l_cv_in = self.input_layers

        # load network output
        qa_mu, qa_logsigma, a = lasagne.layers.get_output(
            [l_qa_mu, l_qa_logsigma, l_qa],
            deterministic=deterministic,
        )
        qz_mu, z = lasagne.layers.get_output(
            [l_qz_mu, l_qz],
            # {l_qz_in : T.zeros_like(qa_mu), l_qa_in : X},
            # {l_qz_in : qa_mu, l_qa_in : X},
            {l_qz_in : a, l_qa_in : X},
            deterministic=deterministic,
        )
        pa_mu, pa_logsigma = lasagne.layers.get_output(
            [l_pa_mu, l_pa_logsigma], z,
            deterministic=deterministic,
        )

        if self.model == 'bernoulli':
            px_mu = lasagne.layers.get_output(
                l_px_mu, z, deterministic=deterministic)
        elif self.model == 'gaussian':
            px_mu, px_logsigma  = lasagne.layers.get_output(
                [l_px_mu, l_px_logsigma], z,
                deterministic=deterministic,
            )

        # entropy term
        log_qa_given_x  = log_normal2(a, qa_mu, qa_logsigma).sum(axis=1)
        log_qz_given_x = log_bernoulli(z, qz_mu).sum(axis=1)
        log_qz_given_x_dgz = log_bernoulli(dg(z), qz_mu).sum(axis=1)
        # log_qz_given_x = log_normal2(z, qz_mu, qz_logsigma).sum(axis=1)
        # log_qz_given_x_dgz = log_normal2(dg(z), qz_mu, qz_logsigma).sum(axis=1)
        log_qza_given_x =  log_qz_given_x + log_qa_given_x

        # log-probability term
        z_prior = T.ones_like(z)*np.float32(0.5)
        log_pz = log_bernoulli(z, z_prior).sum(axis=1)
        # z_prior_sigma = T.cast(T.ones_like(qz_logsigma), dtype=theano.config.floatX)
        # z_prior_mu = T.cast(T.zeros_like(qz_mu), dtype=theano.config.floatX)
        # log_pz = log_normal(z, z_prior_mu,  z_prior_sigma).sum(axis=1)
        log_px_given_z = log_bernoulli(x, px_mu).sum(axis=1)
        log_pa_given_z = log_normal2(a, pa_mu, pa_logsigma).sum(axis=1)

        log_pxz = log_pa_given_z + log_px_given_z + log_pz

        # save them for later
        if deterministic == False:
            self.log_pxz = log_pxz
            self.log_px_given_z = log_px_given_z
            self.log_pz = log_pz
            self.log_qza_given_x = log_qza_given_x
            self.log_qa_given_x = log_qa_given_x
            self.log_qz_given_x = log_qz_given_x
            self.log_qz_given_x_dgz = log_qz_given_x_dgz

        # return log_paxz, log_qza_given_x
        return log_pxz, log_qza_given_x

    def create_objectives(self, deterministic=False):
        # load probabilities
        log_paxz, log_qza_given_x = self._create_components(deterministic=deterministic)

        # compute the evidence lower bound
        elbo = T.mean(log_paxz - log_qza_given_x)
        log_qa_given_x = self.log_qa_given_x

        # we don't use a spearate accuracy metric right now
        return -elbo, T.mean(log_qa_given_x)

    def create_gradients(self, loss, deterministic=False):
        # load networks
        l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
        l_qa_mu, l_qa_logsigma, l_qz_mu, l_qz_logsigma, l_qa, l_qz, l_cv, c, v = self.network

        # load params
        p_params  = lasagne.layers.get_all_params(
            # [l_px_mu], trainable=True)
            [l_px_mu, l_pa_mu, l_pa_logsigma], trainable=True)
        qa_params  = lasagne.layers.get_all_params(l_qa_mu, trainable=True)
        qz_params  = lasagne.layers.get_all_params(l_qz, trainable=True)
        cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)

        # load neural net outputs (probabilities have been precomputed)
        log_pxz, log_px_given_z, log_pz = self.log_pxz, self.log_px_given_z, self.log_pz
        log_qza_given_x = self.log_qza_given_x
        log_qz_given_x = self.log_qz_given_x
        log_qz_given_x_dgz = self.log_qz_given_x_dgz
        cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

        # compute learning signals
        l0 = log_px_given_z + log_pz - log_qz_given_x #- cv # NOTE: this disn't have q(a)
        l_avg, l_var = l0.mean(), l0.var()
        c_new = 0.8*c + 0.2*l_avg
        v_new = 0.8*v + 0.2*l_var
        l = (l0 - c_new) / T.maximum(1, T.sqrt(v_new))
        l_target = (l0 - c_new) / T.maximum(1, T.sqrt(v_new))
        # l_target = log_px_given_z + log_pz - log_qz_given_x

        # compute grad wrt p
        p_grads = T.grad(-log_pxz.mean(), p_params)

        # compute grad wrt q_a
        elbo = T.mean(log_pxz - log_qza_given_x)
        qa_grads = T.grad(-elbo, qa_params)

        # compute grad wrt q_z
        qz_target = T.mean(dg(l_target) * log_qz_given_x_dgz)
        qz_grads = T.grad(-0.2*qz_target, qz_params) # 5x slower rate for q
        # qz_grads = T.grad(-0.2*T.mean(l0), qz_params) # 5x slower rate for q
        # qz_grads = T.grad(-0.2*elbo, qz_params) # 5x slower rate for q

        # compute grad of cv net
        cv_target = T.mean(l0**2)
        # cv_grads = [0.2*g for g in T.grad(cv_target, cv_params)]

        # combine and clip gradients
        clip_grad = 1
        max_norm = 5
        # grads = p_grads + qa_grads + qz_grads + cv_grads
        grads = p_grads + qa_grads + qz_grads #+ cv_grads
        mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
        cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

        return cgrads

    def gen_samples(self, deterministic=False):
        s = self.inputs[-1]
        # put it through the decoder
        _, _, l_px_in, _ = self.input_layers
        l_px_mu = self.network[0]
        px_mu = lasagne.layers.get_output(l_px_mu, {l_px_in : s})

        return px_mu

    def get_params(self):
        # load networks
        l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
        l_qa_mu, l_qa_logsigma, l_qz_mu, l_qz_logsigma, \
        l_qa, l_qz, l_cv, c, v = self.network

        if self.model == 'gaussian':
            raise NotImplementedError('The code below needs to implement Gaussians')

        # load params
        p_params  = lasagne.layers.get_all_params(
            [l_px_mu, l_pa_mu, l_pa_logsigma], trainable=True)
        qa_params  = lasagne.layers.get_all_params(l_qa_mu, trainable=True)
        qz_params  = lasagne.layers.get_all_params(l_qz, trainable=True)
        cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)

        return p_params + qa_params +qz_params # + cv_params

    def create_updates(self, grads, params, alpha, opt_alg, opt_params):
        # call super-class to generate SGD/ADAM updates
        grad_updates = Model.create_updates(self, grads, params, alpha, opt_alg, opt_params)

        # create updates for centering signal
        # load neural net outputs (probabilities have been precomputed)
        l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
        l_qa_mu, l_qa_logsigma, l_qz_mu, l_qz_logsigma, l_qa, l_qz, l_cv, c, v = self.network

        # load neural net outputs (probabilities have been precomputed)
        log_pxz, log_px_given_z, log_pz = self.log_pxz, self.log_px_given_z, self.log_pz
        log_qz_given_x = self.log_qz_given_x
        cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

        # compute learning signals
        l = log_px_given_z + log_pz - log_qz_given_x - cv
        l_avg, l_var = l.mean(), l.var()
        c_new = 0.8*c + 0.2*l_avg
        v_new = 0.8*v + 0.2*l_var

        # compute update for centering signal
        cv_updates = {c : c_new, v : v_new}

        return OrderedDict( grad_updates.items() + cv_updates.items() )
