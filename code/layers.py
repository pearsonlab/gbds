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
        self.s = params['s'].astype(theano.config.floatX)  # standard deviation

        # learnable posterior parameters
        # normal dist over biases
        self.mu = self.add_param(param_init, (num_biases, num_inputs),
                                 name='mu')

        self.log_sig = self.add_param(param_init, (num_biases, num_inputs),
                                      name='log_sig')

        # log normal over rows
        self.alpha = self.add_param(param_init, (num_biases, 1),
                                    name='alpha')
        self.log_beta = self.add_param(param_init, (num_biases, 1),
                                       name='log_beta')
        # standard deviation will always be positive but optimization over
        # log_sig and log_beta can be unconstrained
        self.sigma = T.exp(self.log_sig)
        self.beta = T.exp(self.log_beta)
        self.draw_biases()
        self.draw_on_every_output = True

    def set_mode(self, mode):
        self.mode = mode

    def draw_biases(self):
        self.gamma = self.mu + self.srng.normal(self.sigma.shape) * self.sigma
        self.tau = T.exp(self.alpha + self.srng.normal(self.beta.shape) * self.beta)
        self.tau = T.addbroadcast(self.tau, 1)  # allow broadcasting across columns

    def get_ELBO(self, nbatches):
        """
        Return the contribution to the ELBO for these biases

        Normalized by nbatches (number of batches in dataset)
        """
        # Log Density
        ELBO = (-(self.gamma**2) / (2 * self.tau * self.s**2) -
                0.5 * T.log(2 * np.pi * self.tau * self.s**2)).sum()
        ELBO += ((self.alpha - 1) * T.log(self.tau) + self.a * T.log(self.b) -
                 self.b * self.tau - T.gammaln(self.a)).sum()
        # entropy
        ELBO += (0.5 * T.log(2 * np.pi) + 0.5 + T.log(self.sigma)).sum()
        ELBO += (T.log(self.beta) + self.alpha + 0.5 + 0.5 * T.log(2 * np.pi)).sum()
        return ELBO / nbatches

    def get_output_for(self, input, **kwargs):
        if self.draw_on_every_output:
            self.draw_biases()
        act_biases = self.mode.astype(theano.config.floatX).reshape((1, -1)).dot(self.gamma)

        return input + act_biases
