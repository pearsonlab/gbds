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
from lasagne.nonlinearities import rectify, linear
from layers import PKBiasLayer, PKRowBiasLayer


def get_network(batch_size, input_dim, output_dim, hidden_dim, num_layers,
                PKLparams=None, srng=None, batchnorm=False, is_shooter=False,
                row_sparse=False, add_pklayers=False, filt_size=None,
                hidden_nonlin=rectify, output_nonlin=linear,
                init_std=1.0, add_bias=True):
    """
    Returns a NN with the specified parameters.
    Also returns a list of PKBias layers
    """
    PKbias_layers = []
    NN = lasagne.layers.InputLayer((batch_size, input_dim))
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
            layer_dim = output_dim
            layer_nonlin = output_nonlin
        else:
            layer_dim = hidden_dim
            layer_nonlin = hidden_nonlin

        if batchnorm and i < num_layers - 1 and i != 0:
            if add_bias == False:
                NN = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
                NN,
                layer_dim,
                nonlinearity=layer_nonlin,
                W=lasagne.init.Normal(std=init_std), b=None))
            else:
                NN = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
                NN,
                layer_dim,
                nonlinearity=layer_nonlin,
                W=lasagne.init.Normal(std=init_std)))
        else:
            if add_bias == False:
                NN = lasagne.layers.DenseLayer(
                NN,
                layer_dim,
                nonlinearity=layer_nonlin,
                W=lasagne.init.Normal(std=init_std),b=None)
            else:
                NN = lasagne.layers.DenseLayer(
                NN,
                layer_dim,
                nonlinearity=layer_nonlin,
                W=lasagne.init.Normal(std=init_std))
    if add_pklayers:
        return NN, PKbias_layers
    else:
        return NN
