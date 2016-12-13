import lasagne
import theano.tensor as T
from layers import *
from lasagne.nonlinearities import rectify, linear
from utils import get_network


class DLGM(object):
    """
    Deep Latent Gaussian Model (with tweaks for penaltyshot)

    Rezende, Danilo J., Mohamed, Shakir, and Wierstra, Daan.
    Stochastic backpropagation and approximate inference in deep
    generative models. Technical report, arXiv:1401.4082, 2014.
    """
    def __init__(self, nlayers_gen, nlayers_rec, ninput, nhidden, noutput,
                 srng, k, extra_noise_xi=0.01, extra_noise_s=None, p=None,
                 dropout=False, nonlinearity=rectify,
                 param_init=lasagne.init.Normal(0.01)):
        self.DLGM_layers = []
        self.network = lasagne.layers.InputLayer((None, ninput))
        self.dropout = dropout
        self.srng = srng
        self.p = p  # penalty on noise of output layer
        # scale of added noise to state input as regularization during training
        self.extra_noise_s = extra_noise_s
        if self.dropout:
            self.network = lasagne.layers.DropoutLayer(self.network, p=0.2)
        for i in range(nlayers_gen):
            if i == 0:  # first layer, no nonlinearity on input
                outdims = nhidden
                nonlin = linear
                output_layer = False
            elif i == nlayers_gen - 1:  # last layer, output dim
                outdims = noutput
                nonlin = nonlinearity
                output_layer = True
            else:
                outdims = nhidden
                nonlin = nonlinearity
                output_layer = False
            # recognition networks that convert latent to xi for each layer
            # input is J and output is a vector that parametrizes xi
            # dims of xi are equal to the output dims of that layer
            rec_nets = {'mu_net': get_network(noutput, outdims, nhidden,
                                              nlayers_rec, None, None,
                                              add_pklayers=False),
                        'u_net': get_network(noutput, outdims, nhidden,
                                             nlayers_rec, None, None,
                                             add_pklayers=False),
                        'unc_d_net': get_network(noutput, outdims, nhidden,
                                                 nlayers_rec, None, None,
                                                 add_pklayers=False)}
            self.DLGM_layers.append(DLGMLayer(self.network, outdims, srng,
                                              rec_nets, k,
                                              output_layer=output_layer,
                                              extra_noise=extra_noise_xi,
                                              nonlinearity=nonlin))

            self.network = self.DLGM_layers[-1]

    def get_params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)

    def fuzz_states(self, state):
        """
        Add noise to state input (scaled to range of input)
        """
        state_range = (state.max(axis=0, keepdims=True) -
                       state.min(axis=0, keepdims=True))
        scaled_noise = (state_range * self.extra_noise_s *
                        self.srng.normal(state.shape))
        return state + scaled_noise


    def get_output(self, state, training=False, postJ=None):
        """
        Return a value, J, from the generative model, given a state
        NOTE: This is different from the original DLGM paper where values were
              generated purely randomly.
        """
        if postJ is not None:
            for layer in self.DLGM_layers:
                layer.calculate_xi(postJ)
            use_rec = True
        else:
            use_rec = False

        if self.extra_noise_s is not None:
            state = self.fuzz_states(state)

        if self.dropout:
            return lasagne.layers.get_output(self.network, inputs=state,
                                             deterministic=(not training),
                                             add_noise=training,
                                             use_rec_model=use_rec)
        else:
            return lasagne.layers.get_output(self.network, inputs=state,
                                             add_noise=training,
                                             use_rec_model=use_rec)

    def get_ELBO(self, postJ, Jsamps):
        """
        Get evidence lower bound for this model
        """
        ELBO = 0
        for layer in self.DLGM_layers:
            ELBO += layer.get_ELBO(postJ.shape[0])

        out_layer = self.network
        p_J = 0
        for predJ in Jsamps:
            resJ = postJ - predJ
            p_J -= 0.5 * (T.nlinalg.matrix_inverse(out_layer.G)
                           .dot(resJ.T)**2).sum()
        ELBO += p_J / np.cast[theano.config.floatX](len(Jsamps))
        ELBO -= T.log(T.diag(out_layer.G)).sum()
        if self.p is not None:
            ELBO -= self.p * T.sqrt((out_layer.G**2).sum())
        return ELBO
