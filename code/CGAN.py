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
                 nonlinearity=leaky_rectify, init_std=1.0,
                 condition_noise=None):
        # Neural network (G) that generates data to match the real data
        self.gen_net = get_network(batch_size,
                                   ndims_condition + ndims_noise, ndims_data,
                                   ndims_hidden, nlayers_G,
                                   init_std=init_std,
                                   hidden_nonlin=nonlinearity)
        # Neural network (D) that generates probability of input data being real
        self.discr_net = get_network(batch_size,
                                     ndims_condition + ndims_data, 1,
                                     ndims_hidden, nlayers_D,
                                     init_std=init_std,
                                     hidden_nonlin=nonlinearity,
                                     output_nonlin=sigmoid)
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

    def fuzz_conditions(self, conditions):
        """
        Add noise to conditions (scaled to range)
        """
        conditions_range = (conditions.max(axis=0, keepdims=True) -
                            conditions.min(axis=0, keepdims=True))
        scaled_noise = (conditions_range * self.condition_noise *
                        self.srng.normal(conditions.shape))
        return conditions + scaled_noise

    def get_generated_data(self, conditions, training=False):
        """
        Return generated sample from G given conditions.
        """
        if self.condition_noise is not None and training:
            conditions = self.fuzz_conditions(conditions)
        noise = self.srng.normal((conditions.shape[0], self.ndims_noise))
        inp = T.horizontal_stack(noise, conditions)
        gen_data = lasagne.layers.get_output(self.gen_net, inputs=inp)
        return gen_data

    def get_discr_probs(self, data, conditions, training=False):
        """
        Return probabilities of being real data from discriminator network,
        given conditions
        """
        if self.condition_noise is not None and training:
            conditions = self.fuzz_conditions(conditions)
        inp = T.horizontal_stack(data, conditions)
        discr_probs = lasagne.layers.get_output(self.discr_net, inputs=inp)
        return discr_probs

    def get_gen_params(self):
        return lasagne.layers.get_all_params(self.gen_net)

    def get_discr_params(self):
        return lasagne.layers.get_all_params(self.discr_net)

    def get_discr_cost(self, real_data, fake_data, conditions):
        real_discr_probs = self.get_discr_probs(real_data, conditions,
                                                training=True)
        fake_discr_probs = self.get_discr_probs(fake_data, conditions,
                                                training=True)
        cost = (T.log(real_discr_probs + 1e-12).sum() +
                T.log((1.0 - fake_discr_probs) + 1e-12).sum())
        return cost

    def get_gen_cost(self, gen_data, conditions):
        fake_discr_probs = self.get_discr_probs(gen_data, conditions,
                                                training=True)
        cost = T.log(fake_discr_probs + 1e-12).sum()
        return cost
