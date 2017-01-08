import numpy as np
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# ----------------------------------------------------------------------------

class GumbelSoftmax:
    def __init__(self, tau, eps=1e-20):
        assert tau != 0
        self.temperature=tau
        self.eps=eps
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

    def __call__(self, logits):
        #sample from Gumbel(0, 1)
        uniform = self._srng.uniform(logits.shape,low=0,high=1)
        gumbel = -T.log(-T.log(uniform + self.eps) + self.eps)

        #draw a sample from the Gumbel-Softmax distribution
        return T.nnet.softmax((logits + gumbel) / self.temperature)


def onehot_argmax(logits):
    return T.extra_ops.to_one_hot(T.argmax(logits,-1),logits.shape[-1])


class GumbelSoftmaxSampleLayer(lasagne.layers.Layer):
    def __init__(self, incoming, tau, eps=1e-20, **kwargs):
        super(GumbelSoftmaxSampleLayer, self).__init__(incoming, **kwargs)
        self.gumbel_softmax = GumbelSoftmax(tau, eps=eps)

    def get_output_for(self, input, hard_max=False, **kwargs):
        if hard_max:
            return onehot_argmax(input)
        else:
            return self.gumbel_softmax(input)


class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)


class BernoulliSampleLayer(lasagne.layers.Layer):
    def __init__(self, mean,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(BernoulliSampleLayer, self).__init__(mean, **kwargs)
        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
        self._srng.seed(seed)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, mu, **kwargs):
        return self._srng.binomial(size=mu.shape, p=mu, dtype=mu.dtype)
