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
import theano
import lasagne
import theano.tensor as T
import numpy as np


class DLGMLayer(lasagne.layers.Layer):
    """
    This layer is inspired by the paper "Stochastic Backpropagation and
    Approximate Inference in Deep Generative Models"

    incoming (Lasagne Layer): preceding layer in DLGM
    num_units (int): number of output units in this layer
    srng (theano RandomState): random number generator
    rec_nets (dictionary of lasagne NNs): Neural networks that
        paramaterize the recognition model
    J (theano symbolic matrix): Input to rec model
    k (float): regularization term on generative weights
    """
    def __init__(self, incoming, num_units, srng, rec_nets, k,
                 output_layer=False, extra_noise=0.01,
                 param_init=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(DLGMLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.srng = srng
        self.num_units = num_units
        self.output_layer = output_layer
        self.extra_noise = extra_noise

        # Initialize generative/decoding Parameters
        self.W = self.add_param(param_init, (num_inputs, num_units),
                                name='W')
        self.b = self.add_param(param_init, (num_units,), name='b')
        self.unc_G = self.add_param(param_init, (num_units, num_units),
                                    name='unc_G')
        self.G = (T.diag(T.nnet.softplus(T.diag(self.unc_G))) +
                  T.tril(self.unc_G, k=-1))
        self.nonlinearity = nonlinearity

        # regularization term
        self.k = k

        # Load recognition/encoding Parameters
        self.mu_net = rec_nets['mu_net']
        self.u_net = rec_nets['u_net']
        self.unc_d_net = rec_nets['unc_d_net']

        # add parameters to layer class
        rec_params = (lasagne.layers.get_all_params(self.mu_net) +
                      lasagne.layers.get_all_params(self.u_net) +
                      lasagne.layers.get_all_params(self.unc_d_net))
        for param in rec_params:
            self.add_param(param, param.shape.eval())

    def calculate_xi(self, postJ):
        """
        Calculate xi based on sampled J from posterior
        """
        # get output of rec model
        self.batch_mu = lasagne.layers.get_output(self.mu_net, inputs=postJ)
        self.batch_u = lasagne.layers.get_output(self.u_net, inputs=postJ)
        self.batch_unc_d = lasagne.layers.get_output(self.unc_d_net,
                                                     inputs=postJ)

        # add extra dim to batch_u, so it gets treated as column vectors when
        # iterated over
        self.batch_u = self.batch_u.reshape(
            (self.batch_u.shape[0], self.batch_u.shape[1], 1))

        def get_cov(u, unc_d):
            # convert output of rec model to rank-1 covariance matrix

            # use softplus to get positive constrained d, minimum of -15
            # since softplus will turn low numbers into 0, which become NaNs
            # when inverted
            d = T.nnet.softplus(T.maximum(unc_d, -15))
            D_inv = T.diag(1.0 / d)
            eta = 1.0 / (u.T.dot(D_inv).dot(u) + 1.0)
            C = D_inv - eta * D_inv.dot(u).dot(u.T).dot(D_inv)
            Tr_C = T.nlinalg.trace(C)
            ld_C = T.log(eta) - T.log(d).sum()  # eq 20 in DLGM
            # coeff = ((1 - T.sqrt(eta)) / (u.T.dot(D_inv).dot(u)))
            # simplified coefficient below is more stable as u -> 0
            # original coefficient from paper is above
            coeff = eta / (1 + T.sqrt(eta))
            R = T.sqrt(D_inv) - coeff * D_inv.dot(u).dot(u.T).dot(T.sqrt(D_inv))
            return Tr_C, ld_C, R

        (self.batch_Tr_C, self.batch_ld_C, self.batch_R), _ = theano.scan(
            fn=get_cov, outputs_info=None, sequences=[self.batch_u,
                                                      self.batch_unc_d])
        self.batch_xi = (self.batch_mu +
                         T.batched_dot(self.batch_R,
                                       self.srng.normal(
                                           (self.batch_R.shape[0],
                                            self.num_units))))

    def get_ELBO(self, length):
        """
        Get ELBO for this layer

        length (theano symbolic int): length of current batch
        """
        #  KL divergence between posterior and N(0,1) prior
        KL_div = 0.5 * (T.sqrt((self.batch_mu**2).sum(axis=1)).sum() +
                        self.batch_Tr_C.sum() - self.batch_ld_C.sum() -
                        length)
        weight_reg = ((0.5 / self.k) *
                      T.sqrt((self.W**2).sum()) *
                      T.sqrt((self.G**2).sum()))
        return -(weight_reg + KL_div)

    def get_output_for(self, input, add_noise=False, use_rec_model=False,
                       **kwargs):
        activation = self.nonlinearity(input).dot(self.W) + self.b
        if use_rec_model:
            # use sample from rec model
            xi = self.batch_xi
            if add_noise:  # additional noise
                xi += self.extra_noise * self.srng.normal(self.batch_xi.shape)
        else:
            # pure random input
            xi = self.srng.normal((input.shape[0], self.num_units))
        # we want the mean when training, so don't add noise to
        # output of last layer when training.
        if not self.output_layer:
            activation += T.dot(xi, self.G)
        elif not add_noise:
            activation += T.dot(xi, self.G)
        return activation

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class PKBiasLayer(lasagne.layers.Layer):
    """
    This layer draws different biases (depending on the mode)
    from a normal distribution, then adds them to the input

    Default modes are as follows:
    0: normal, no biases added
    1: saline and DLPFC, bias 0 is added
    2: saline and DMPFC, bias 1 is added
    3: muscimol and DLPFC, biases 0 and 2 are added
    4: muscimol and DMPFC, biases 1 and 3 are added
    """
    def __init__(self, incoming, srng, params, param_init=lasagne.init.Normal(0.01),
                 num_biases=4, **kwargs):
        super(PKBiasLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.mode = T.zeros(num_biases)
        self.srng = srng
        self.k = np.cast[theano.config.floatX](params['k'])

        self.m = self.add_param(param_init, (num_biases, num_inputs),
                                name='m')
        self.log_s = self.add_param(param_init, (num_biases, num_inputs),
                                    name='log_s')
        # standard deviation will always be positive but optimization over
        # log_s can be unconstrained
        self.s = T.exp(self.log_s)
        self.draw_biases()
        self.draw_on_every_output = True

    def set_mode(self, mode):
        self.mode = mode

    def draw_biases(self):
        self.biases = self.m + self.srng.normal(self.s.shape) * self.s

    def get_ELBO(self, nbatches):
        """
        Return the contribution to the ELBO for these biases

        Normalized by nbatches (number of batches in dataset)
        """
        ELBO = (-T.abs_(self.biases) / self.k - T.log(2 * self.k)).sum()
        ELBO += T.log(self.s).sum()
        return ELBO / nbatches

    def get_output_for(self, input, **kwargs):
        if self.draw_on_every_output:
            self.draw_biases()
        act_biases = self.mode.astype(theano.config.floatX).reshape((1, -1)).dot(self.biases)

        return input + act_biases


class PKRowBiasLayer(lasagne.layers.Layer):
    """
    This layer draws different biases (depending on the mode)
    from a normal distribution, then adds them to the input.
    This layer has sparsity at the row level, instead of the individual
    sparsity of the PKBiasLayer.

    Default modes are as follows:
    0: normal, no biases added
    1: saline and DLPFC, bias 0 is added
    2: saline and DMPFC, bias 1 is added
    3: muscimol and DLPFC, biases 0 and 2 are added
    4: muscimol and DMPFC, biases 1 and 3 are added
    """
    def __init__(self, incoming, srng, params, param_init=lasagne.init.Normal(0.01),
                 num_biases=4, **kwargs):
        super(PKRowBiasLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.mode = T.zeros(num_biases)
        self.srng = srng
        # parameters on prior
        self.a = np.cast[theano.config.floatX](params['a'])  # shape
        self.b = np.cast[theano.config.floatX](params['b'])  # rate

        # learnable posterior parameters
        # normal dist over biases
        self.mu = self.add_param(param_init, (num_biases, num_inputs),
                                 name='mu')

        self.unc_sig = self.add_param(param_init, (num_biases, num_inputs),
                                      name='unc_sig')

        # gamma over rows
        self.alpha = theano.shared(value=self.a * np.ones((num_biases, 1)),
                                   name='alpha', broadcastable=[False, True])
        self.beta = theano.shared(value=self.b * np.ones((num_biases, 1)),
                                  name='beta', broadcastable=[False, True])

        # update for alpha
        self.alpha += (num_inputs / 2.0)

        # standard deviation will always be positive but optimization over
        # unc_sig can be unconstrained
        self.sigma = T.nnet.softplus(self.unc_sig)
        self.draw_biases()
        self.draw_on_every_output = True

    def set_mode(self, mode):
        self.mode = mode

    def draw_biases(self):
        self.gamma = self.mu + self.srng.normal(self.sigma.shape) * self.sigma

    def coord_update(self):
        self.beta = self.b + 0.5 * (self.mu**2 + self.sigma**2).sum(axis=1,
                                                                    keepdims=True)
        self.beta = T.addbroadcast(self.beta, 1)

    def get_ELBO(self, nbatches):
        """
        Return the contribution to the ELBO for these biases

        Normalized by nbatches (number of batches in dataset)
        """
        self.coord_update()
        # Log Density
        ELBO = (-0.5 * (self.mu**2 + self.sigma**2) * (self.alpha / self.beta) +
                0.5 * (T.psi(self.alpha) - T.log(self.beta)) -
                0.5 * T.log(2 * np.pi)).sum()
        ELBO += ((self.a - 1) * (T.psi(self.alpha) - T.log(self.beta)) -
                 self.b * (self.alpha / self.beta) + self.a * T.log(self.b) -
                 T.gammaln(self.a)).sum()
        # entropy
        ELBO += (0.5 * T.log(2 * np.pi) + 0.5 + T.log(self.sigma)).sum()
        ELBO += (self.alpha - T.log(self.beta) + T.gammaln(self.alpha) +
                 (1 - self.alpha) * T.psi(self.alpha)).sum()
        return ELBO / nbatches

    def get_output_for(self, input, **kwargs):
        if self.draw_on_every_output:
            self.draw_biases()
        act_biases = self.mode.astype(theano.config.floatX).reshape((1, -1)).dot(self.gamma)

        return input + act_biases
