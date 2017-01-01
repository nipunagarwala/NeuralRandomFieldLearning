import numpy as np
import theano.tensor as T
import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.gradient import consider_constant

# ----------------------------------------------------------------------------

class GumbelSoftmaxSampleLayer(lasagne.layers.Layer):
    def __init__(self, mean,
                 temperature=1, hard=False,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        """
        Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
        """
        super(GumbelSoftmaxSampleLayer, self).__init__(mean, **kwargs)

        # save sample parameters
        self._srng       = RandomStreams(seed)
        self.temperature = temperature
        self.hard        = hard

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
        self._srng.seed(seed)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def _sample_gumbel(self, shape, eps=1e-20):
        """ Sample from Gumbel(0, 1) """
        U = self._srng.uniform(size=shape, low=0, high=1)
        return -T.log(-T.log(U + eps) + eps)

    def _sample_gumbel_softmax(self, logits, temperature):
        """ Sample from Gumbel-Softmax distribution """
        y = logits + self._sample_gumbel(T.shape(logits))
        return T.nnet.softmax(y / temperature)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        y = self._sample_gumbel_softmax(inputs, self.temperature)
        if self.hard:
            k = T.shape(inputs)[-1]
            y_hard = T.cast(T.equal(y, T.argmax(y, axis=1, keep_dims=True)), y.dtype)
            y = consider_constant(y_hard - y) + y
        return y

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
