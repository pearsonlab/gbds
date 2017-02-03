import theano
import lasagne
import theano.tensor as T
import numpy as np
from nn_utils import get_network
from lasagne.nonlinearities import sigmoid, leaky_rectify


class CGAN(object):
    """
    Conditional Generative Adversarial Network

    Mirza, Mehdi, Osindero, Simon, Conditional generative adversarial nets,
    Arxiv:1411.1784v1, 2014
    """
    def __init__(self, nlayers_G, nlayers_D, ndims_condition, ndims_noise,
                 ndims_hidden, ndims_data, batch_size, srng,
                 nonlinearity=leaky_rectify, init_std_G=1.0,
                 init_std_D=0.005,
                 condition_noise=None, condition_scale=None,
                 instance_noise=None):
        # Neural network (G) that generates data to match the real data
        self.gen_net = get_network(batch_size,
                                   ndims_condition + ndims_noise, ndims_data,
                                   ndims_hidden, nlayers_G,
                                   init_std=init_std_G,
                                   hidden_nonlin=nonlinearity,
                                   batchnorm=True)
        # Neural network (D) that generates probability of input data being real
        self.discr_net = get_network(batch_size,
                                     ndims_condition + ndims_data, 1,
                                     ndims_hidden, nlayers_D,
                                     init_std=init_std_D,
                                     hidden_nonlin=nonlinearity,
                                     batchnorm=True)
        # size of minibatches (number of rows)
        self.batch_size = batch_size
        # symbolic random number generator
        self.srng = srng
        # min and max of noise samples
        self.noise_bounds = (-1, 1)
        # number of dimensions of conditional input
        self.ndims_condition = ndims_condition
        # number of dimensions of noise input
        self.ndims_noise = ndims_noise
        # number of hidden units
        self.ndims_hidden = ndims_hidden
        # number of dimensions in the data
        self.ndims_data = ndims_data
        # scale of added noise to conditions as regularization during training
        self.condition_noise = condition_noise
        # scale of each condition dimension, used for normalization
        self.condition_scale = condition_scale
        # scale of added noise to data input into discriminator
        # http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        self.instance_noise = instance_noise

    def get_generated_data(self, conditions, training=False):
        """
        Return generated sample from G given conditions.
        """
        if self.condition_scale is not None:
            conditions /= self.condition_scale
        if self.condition_noise is not None and training:
            conditions += (self.condition_noise *
                           self.srng.normal(conditions.shape))
        noise = 2 * self.srng.uniform((conditions.shape[0],
                                       self.ndims_noise)) - 1
        # noise = self.srng.normal((conditions.shape[0],
        #                           self.ndims_noise))
        inp = T.horizontal_stack(noise, conditions)
        gen_data = lasagne.layers.get_output(self.gen_net, inputs=inp,
                                             deterministic=(not training))
        return gen_data

    def get_discr_probs(self, data, conditions, training=False):
        """
        Return probabilities of being real data from discriminator network,
        given conditions
        """
        if self.condition_scale is not None:
            conditions /= self.condition_scale
        if self.condition_noise is not None and training:
            conditions += (self.condition_noise *
                           self.srng.normal(conditions.shape))
        if self.instance_noise is not None and training:
            data += (self.instance_noise *
                     self.srng.normal((data.shape)))
        inp = T.horizontal_stack(data, conditions)
        discr_probs = lasagne.layers.get_output(self.discr_net, inputs=inp,
                                                deterministic=(not training))
        return discr_probs

    def get_gen_params(self):
        return lasagne.layers.get_all_params(self.gen_net, trainable=True)

    def get_discr_params(self):
        return lasagne.layers.get_all_params(self.discr_net, trainable=True)

    def get_discr_cost(self, real_data, fake_data, conditions):
        real_discr_probs = self.get_discr_probs(real_data, conditions,
                                                training=True)
        fake_discr_probs = self.get_discr_probs(fake_data, conditions,
                                                training=True)
        cost = real_discr_probs.mean() - fake_discr_probs.mean()
        return cost

    def get_gen_cost(self, gen_data, conditions):
        fake_discr_probs = self.get_discr_probs(gen_data, conditions,
                                                training=True)
        cost = fake_discr_probs.mean()
        return cost
