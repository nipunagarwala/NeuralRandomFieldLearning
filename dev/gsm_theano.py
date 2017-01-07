import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
import sys; sys.path.append('../models/layers')
from sampling import GumbelSoftmaxSampleLayer
sys.path.append('../models/distributions')
from distributions import log_bernoulli

n_class = 10  # number of classes
n_cat = 30  # number of categorical distributions
n_chan = 1  # number of channels
n_dim = 28  # number of dimensions
n_in = n_out = n_chan*n_dim*n_dim  # input/output size

# shared input/training variables
x = T.matrix(dtype=theano.config.floatX)
tau = theano.shared(
    np.float32(1.0), name='temperature',
    allow_downcast=True, borrow=False,
)
lr = theano.shared(
    np.float32(0.001), name='learning_rate',
    allow_downcast=True, borrow=False,
)

# encoder design
net = InputLayer((None, n_in), x)
net = DenseLayer(net, 512, nonlinearity=T.nnet.relu, W=lasagne.init.GlorotUniform())
net = DenseLayer(net, 256, nonlinearity=T.nnet.relu, W=lasagne.init.GlorotUniform())
# bottleneck design
logits_y = DenseLayer(net, n_cat*n_class, nonlinearity=None, W=lasagne.init.GlorotUniform())
logits_y = reshape(logits_y, (-1, n_class))
y = GumbelSoftmaxSampleLayer(logits_y, tau)
y = reshape(y, (-1, n_cat, n_class))
# decoder design
net = DenseLayer(flatten(y), 256, nonlinearity=T.nnet.relu, W=lasagne.init.GlorotUniform())
net = DenseLayer(net, 512, nonlinearity=T.nnet.relu, W=lasagne.init.GlorotUniform())
logits_x = DenseLayer(net, n_out, nonlinearity=None, W=lasagne.init.GlorotUniform())

# define the loss
_logits_y, _logits_x = lasagne.layers.get_output([logits_y, logits_x], deterministic=False)
q_y = T.nnet.softmax(_logits_y)
log_q_y = T.log(q_y + 1e-20)
log_p_x = log_bernoulli(x, _logits_x)

kl_tmp = T.reshape(q_y * (log_q_y - T.log(1.0 / n_class)), [-1 , n_cat, n_class])
KL = T.sum(kl_tmp, axis=[1, 2])
elbo = T.sum(log_p_x, 1) - KL
loss = T.mean(-elbo)

# network compilation
params = get_all_params(logits_x, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=lr)
train_op = theano.function([x], loss, updates=updates)
eval_op = theano.function([x], loss)

# training
BATCH_SIZE = 100
NUM_ITERS = 50000
tau0 = 1.0  # initial temperature
np_temp = tau0
np_lr = 0.001
ANNEAL_RATE = 0.00003
MIN_TEMP = 0.5

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('/tmp/', one_hot=True).train

tau.set_value(np_temp, borrow=False)
lr.set_value(np_lr, borrow=False)
for i in range(1, NUM_ITERS):
    sample, _ = data.next_batch(BATCH_SIZE)
    np_loss = train_op(sample)

    # anneal temp and learning rate
    if i % 1000 == 1:
        np_temp = np.maximum(tau0*np.exp(-ANNEAL_RATE*i), MIN_TEMP)
        np_lr *= 0.9
        tau.set_value(np_temp, borrow=False)
        lr.set_value(np_lr, borrow=False)

    # print log statement
    if i % 5000 == 1:
        _np_loss = eval_op(sample)
        print('Step %d, ELBO: %0.3f' % (i, -_np_loss))
