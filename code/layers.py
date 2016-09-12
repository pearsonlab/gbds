import theano
import lasagne
import theano.tensor as T
import numpy as np


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
