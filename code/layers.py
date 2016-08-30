import theano
import lasagne
import theano.tensor as T


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
    def __init__(self, incoming, srng, param_init=lasagne.init.Normal(0.01),
                 num_biases=4, **kwargs):
        super(PKBiasLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.mode = T.zeros(num_biases)
        self.srng = srng

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

    def get_output_for(self, input, **kwargs):
        if self.draw_on_every_output:
            self.draw_biases()
        act_biases = self.mode.astype(theano.config.floatX).reshape((1, -1)).dot(self.biases)

        return input + act_biases
