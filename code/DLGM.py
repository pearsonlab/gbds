import lasagne
import theano.tensor as T
from layers import *
from lasagne.nonlinearities import rectify, linear

def get_network(input_dim, output_dim, hidden_dim, num_layers,
                PKLparams, srng, batchnorm=False, is_shooter=False,
                row_sparse=False, add_pklayers=False, filt_size=None):
    """
    Returns a NN with the specified parameters.
    Also returns a list of PKBias layers
    """
    PKbias_layers = []
    NN = lasagne.layers.InputLayer((None, input_dim))
    if batchnorm:
        NN = lasagne.layers.BatchNormLayer(NN)
    if filt_size is not None:  # first layer convolution
        # rearrange dims for convolution
        NN = lasagne.layers.DimshuffleLayer(NN, ('x', 1, 0))
        # custom pad so that no timepoint gets input from future
        NN = lasagne.layers.PadLayer(NN, [(0, 0), (filt_size - 1, 0)], batch_ndim=1)
        # Perform convolution (no pad, no filter flip (for interpretability))
        NN = lasagne.layers.Conv1DLayer(NN, num_filters=hidden_dim, filter_size=filt_size,
                                        pad='valid', flip_filters=False,
                                        nonlinearity=rectify)
        # rearrange dims for dense layers
        NN = lasagne.layers.DimshuffleLayer(NN, (2, 1))
    for i in range(num_layers):
        if is_shooter and add_pklayers:
            if row_sparse:
                PK_bias = PKRowBiasLayer(NN, srng, PKLparams)
            else:
                PK_bias = PKBiasLayer(NN, srng, PKLparams)
            PKbias_layers.append(PK_bias)
            NN = PK_bias
        if i == num_layers - 1:
            NN = lasagne.layers.DenseLayer(NN,
                                           output_dim,
                                           nonlinearity=linear,
                                           W=lasagne.init.Normal(std=0.1))
        else:
            NN = lasagne.layers.DenseLayer(NN,
                                           hidden_dim,
                                           nonlinearity=rectify,
                                           W=lasagne.init.Orthogonal())
    if add_pklayers:
        return NN, PKbias_layers
    else:
        return NN


class DLGM(object):
    """
    Deep Latent Gaussian Model (with tweaks for penaltyshot)

    Rezende, Danilo J., Mohamed, Shakir, and Wierstra, Daan.
    Stochastic backpropagation and approximate inference in deep
    generative models. Technical report, arXiv:1401.4082, 2014.
    """
    def __init__(self, nlayers_gen, nlayers_rec, ninput, nhidden, noutput,
                 srng, k, p=None, dropout=False):
        self.DLGM_layers = []
        self.network = lasagne.layers.InputLayer((None, ninput))
        self.dropout = dropout
        self.p = p  # penalty on noise of output layer
        if self.dropout:
            self.network = lasagne.layers.DropoutLayer(self.network, p=0.2)
        for i in range(nlayers_gen):
            if i == 0:  # first layer, no nonlinearity on input
                outdims = nhidden
                nonlin = lasagne.nonlinearities.linear
                output_layer = False
            elif i == nlayers_gen - 1:  # last layer, output dim
                outdims = noutput
                nonlin = lasagne.nonlinearities.rectify
                output_layer = True
            else:
                outdims = nhidden
                nonlin = lasagne.nonlinearities.rectify
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
                                              nonlinearity=nonlin))

            self.network = self.DLGM_layers[-1]

    def get_params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)

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
        if self.dropout:
            return lasagne.layers.get_output(self.network, inputs=state,
                                             deterministic=(not training),
                                             add_noise=training,
                                             use_rec_model=use_rec)
        else:
            return lasagne.layers.get_output(self.network, inputs=state,
                                             add_noise=training,
                                             use_rec_model=use_rec)

    def get_ELBO(self, postJ, predJ):
        """
        Get evidence lower bound for this model
        """
        ELBO = 0
        for layer in self.DLGM_layers:
            ELBO += layer.get_ELBO(postJ.shape[0])

        out_layer = self.network
        resJ = postJ - predJ
        ELBO -= 0.5 * T.sqrt((T.nlinalg.matrix_inverse(out_layer.G)
                              .dot(resJ.T)**2).sum())
        ELBO -= 0.5 * T.log(2 * np.pi)
        ELBO -= T.log(T.diag(out_layer.G)).sum()
        if self.p is not None:
            ELBO -= self.p * T.abs_(out_layer.G).sum()
        return ELBO
