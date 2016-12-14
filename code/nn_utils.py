import lasagne
from lasagne.nonlinearities import rectify, linear
from layers import PKBiasLayer, PKRowBiasLayer


def get_network(input_dim, output_dim, hidden_dim, num_layers,
                PKLparams=None, srng=None, batchnorm=False, is_shooter=False,
                row_sparse=False, add_pklayers=False, filt_size=None,
                hidden_nonlin=rectify, output_nonlin=linear,
                init_std=1.0):
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
                                        nonlinearity=hidden_nonlin)
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
                                           nonlinearity=output_nonlin,
                                           W=lasagne.init.Normal(std=init_std))
        else:
            NN = lasagne.layers.DenseLayer(NN,
                                           hidden_dim,
                                           nonlinearity=hidden_nonlin,
                                           W=lasagne.init.Orthogonal())
    if add_pklayers:
        return NN, PKbias_layers
    else:
        return NN
