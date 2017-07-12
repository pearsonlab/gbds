"""
The MIT License (MIT)
Copyright (c) 2017 Shariq Iqbal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import lasagne
import theano.tensor as T
from nn_utils import get_network
from lasagne.nonlinearities import leaky_rectify


class CGAN(object):
    """
    Conditional Generative Adversarial Network

    Uses Wasserstein GAN formulation

    Mirza, Mehdi, Osindero, Simon, Conditional generative adversarial nets,
    Arxiv:1411.1784v1, 2014

    ndims_condition is all of the conditions minus noise and minus subID!
    """
    def __init__(self, nlayers_G, nlayers_D, ndims_condition, ndims_noise, ndims_hidden, ndims_data, batch_size,srng,
                 nonlinearity=leaky_rectify, init_std_G=1.0,
                 init_std_D=0.005,
                 condition_noise=None, condition_scale=None,
                 instance_noise=None):
        #import pdb; pdb.set_trace()
        #if nlayers_C == None:
        #    self.compressbool = False
        #else: self.compressbool = True

        #if compressbool:
        #    ndims_condcompress = ndims_condition + ndims_compress #6+2+10(compress)
        # Neural network (G) that generates data to match the real data
        #if compressbool:
        #    self.gen_net = get_network(batch_size,
        #                           ndims_condcompress + ndims_noise, ndims_data,
        #                           ndims_hidden, nlayers_G,
        #                           init_std=init_std_G,
        #                           hidden_nonlin=nonlinearity,
        #                           batchnorm=True)
        # elif ndims_subIDcond:
        #     self.gen_net = get_network(batch_size, ndims_condition+ndims_subIDcond+ndims_noise, ndims_data, ndims_hidden, nlayers_G, init_std=init_std_G,
        #                                hidden_nonlin=nonlinearity, batchnorm=True)
        # else: #if this is the gan g0
        self.gen_net = get_network(batch_size, ndims_condition+ndims_noise, ndims_data, ndims_hidden, nlayers_G, init_std=init_std_G,
                                    hidden_nonlin=nonlinearity, batchnorm=True)
        # Neural network (D) that discriminates between real and generated data
        # if compressbool:
        #     self.discr_net = get_network(batch_size,
        #                              ndims_subIDcond + ndims_condition + ndims_data, 1,
        #                              ndims_hidden, nlayers_D,
        #                              init_std=init_std_D,
        #                              hidden_nonlin=nonlinearity,
        #                              batchnorm=True)
        # elif ndims_subIDcond:
        #     self.discr_net = get_network(batch_size, ndims_subIDcond + ndims_condition+ndims_data, 1, ndims_hidden, nlayers_D, init_std=init_std_D, hidden_nonlin=nonlinearity, batchnorm=True)
        # else: #if this is the gan g0
        self.discr_net = get_network(batch_size, ndims_condition+ndims_data, 1, ndims_hidden, nlayers_D, init_std=init_std_D, hidden_nonlin=nonlinearity, batchnorm=True)
        # Neural network (C) that compresses subjectID one-hot-encoded representation. 
        # if compressbool:
        #     self.compress_net = get_network(batch_size, ndims_subIDcond, ndims_compress, 
        #                                     1, nlayers_C, init_std=1.0, hidden_nonlin=nonlinearity, batchnorm=False, add_bias=False)
        # size of minibatches (number of rows)
        self.batch_size = batch_size
        # symbolic random number generator
        self.srng = srng
        # number of dimensions of conditional input without subID
        self.ndims_condition = ndims_condition
        # number of dimensions of subID
        #if self.compressbool:
        #self.ndims_subIDcond = ndims_subIDcond
        # number of dimensions of final output compress layer
        # if compressbool:
        #     self.ndims_compress = ndims_compress
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
        # self.gamma = gamma
        # self.improveWGAN = improveWGAN
        # self.lmbda = lmbda
        # self.compressbool = compressbool

    def get_generated_data(self, conditions, subIDconds=None, training=False):   ### add subID argument
        """
        Return generated sample from G given conditions.
        """
        #import pdb; pdb.set_trace()
        if self.condition_scale is not None:
            conditions /= self.condition_scale
        if self.condition_noise is not None and training:
            conditions += (self.condition_noise *
                           self.srng.normal(conditions.shape))
        #####

        # if self.compressbool:            
        #     compress = lasagne.layers.get_output(self.compress_net, inputs=subIDconds, deterministic=(not training))
        #     conditions = T.horizontal_stack(conditions, compress)
        # elif subIDconds:
        #     conditions = T.horizontal_stack(conditions, subIDconds)
        # else: #if gang0
        #conditions = conditions
        ######

        noise = 2 * self.srng.uniform((conditions.shape[0],
                                       self.ndims_noise)) - 1
        # noise = self.srng.normal((conditions.shape[0],
        #                           self.ndims_noise))
        inp = T.horizontal_stack(noise, conditions)
        gen_data = lasagne.layers.get_output(self.gen_net, inputs=inp,
                                             deterministic=(not training))
        return gen_data

    def get_discr_vals(self, data, conditions, subIDconds=None, training=False):  ### add subID argument
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

        #####
        # if subIDconds:
        #conditions = T.horizontal_stack(conditions, subIDconds)
        ######

        inp = T.horizontal_stack(data, conditions)
        discr_probs = lasagne.layers.get_output(self.discr_net, inputs=inp,
                                                deterministic=(not training))
        return discr_probs

    def get_gen_params(self):
        # if self.compressbool:
        #     return lasagne.layers.get_all_params(self.gen_net, trainable=True) + lasagne.layers.get_all_params(self.compress_net, trainable=True) ### + same command but for compress_net
        # else:
        return lasagne.layers.get_all_params(self.gen_net, trainable=True)

    def get_discr_params(self):
        return lasagne.layers.get_all_params(self.discr_net, trainable=True)

    def get_discr_cost(self, real_data, fake_data, conditions, subID=None):
        #if subID:
        #real_discr_out = self.get_discr_vals(real_data, conditions, subID,
        #                                     training=True)
        #else:
        real_discr_out = self.get_discr_vals(real_data, conditions, training=True)

        # if subID:
        #     fake_discr_out = self.get_discr_vals(fake_data, conditions, subID,
        #                                      training=True)
        # else:
        fake_discr_out = self.get_discr_vals(fake_data, conditions, training=True)
        cost = real_discr_out.mean() - fake_discr_out.mean()

        #  Gradient penalty from "Improved Training of Wasserstein GANs"
        # if self.improveWGAN:
        #     alpha = self.srng.uniform((self.batch_size, 1))
        #     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        #     if subID:
        #         interp_discr_out = self.get_discr_vals(interpolates, conditions, subID,
        #                                            training=True)
        #     else:
        #         interp_discr_out = self.get_discr_vals(interpolates, condition, training=True)
        #     gradients = T.grad(interp_discr_out.sum(), interpolates)
        #     slopes = T.sqrt((gradients**2).sum(axis=1))  # gradient norms
        #     gradient_penalty = T.mean((slopes - 1)**2)
        #     cost -= self.lmbda * gradient_penalty

        return cost

    def get_gen_cost(self, gen_data, conditions, subID=None):
        # if subID:
        #     fake_discr_out = self.get_discr_vals(gen_data, conditions, subID, training=True)
        # else:
        fake_discr_out = self.get_discr_vals(gen_data, conditions, training=True)
        # if self.compressbool: #for the compress_net, we want an L2 penalty
        #     if self.gamma is not None:
        #         penalty = self.gamma * T.sqrt(((lasagne.layers.get_all_params(self.compress_net)[0])**2).sum()) #l2 penalty for compress net
        #     else: 
        #         penalty = 0
        #     cost = fake_discr_out.mean() - penalty
        # else:
        cost = fake_discr_out.mean()
        return cost


class WGAN(object):
    """
    Wasserstein Generative Adversarial Network
    """
    def __init__(self, nlayers_G, nlayers_D, ndims_noise,
                 ndims_hidden, ndims_data, batch_size, srng,
                 nonlinearity=leaky_rectify, init_std_G=1.0,
                 init_std_D=0.005,
                 instance_noise=None):
        # Neural network (G) that generates data to match the real data
        self.gen_net = get_network(batch_size,
                                   ndims_noise, ndims_data,
                                   ndims_hidden, nlayers_G,
                                   init_std=init_std_G,
                                   hidden_nonlin=nonlinearity,
                                   batchnorm=True)
        # Neural network (D) that discriminates between real and generated data
        self.discr_net = get_network(batch_size,
                                     ndims_data, 1,
                                     ndims_hidden, nlayers_D,
                                     init_std=init_std_D,
                                     hidden_nonlin=nonlinearity,
                                     batchnorm=True)
        # size of minibatches (number of rows)
        self.batch_size = batch_size
        # symbolic random number generator
        self.srng = srng
        # number of dimensions of noise input
        self.ndims_noise = ndims_noise
        # number of hidden units
        self.ndims_hidden = ndims_hidden
        # number of dimensions in the data
        self.ndims_data = ndims_data
        # scale of added noise to data input into discriminator
        # http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        self.instance_noise = instance_noise

    def get_generated_data(self, N, training=False):
        """
        Return N generated samples from G.
        """
        noise = 2 * self.srng.uniform((N, self.ndims_noise)) - 1
        # noise = self.srng.normal((N, self.ndims_noise))
        gen_data = lasagne.layers.get_output(self.gen_net, inputs=noise,
                                             deterministic=(not training))
        return gen_data

    def get_discr_vals(self, data, training=False):
        """
        Return probabilities of being real data from discriminator network
        """
        if self.instance_noise is not None and training:
            data += (self.instance_noise *
                     self.srng.normal((data.shape)))
        discr_probs = lasagne.layers.get_output(self.discr_net, inputs=data,
                                                deterministic=(not training))
        return discr_probs

    def get_gen_params(self):
        return lasagne.layers.get_all_params(self.gen_net, trainable=True)

    def get_discr_params(self):
        return lasagne.layers.get_all_params(self.discr_net, trainable=True)

    def get_discr_cost(self, real_data, fake_data):
        real_discr_out = self.get_discr_vals(real_data,
                                             training=True)
        fake_discr_out = self.get_discr_vals(fake_data,
                                             training=True)
        cost = real_discr_out.mean() - fake_discr_out.mean()
        return cost

    def get_gen_cost(self, gen_data):
        fake_discr_out = self.get_discr_vals(gen_data,
                                             training=True)
        cost = fake_discr_out.mean()
        return cost
