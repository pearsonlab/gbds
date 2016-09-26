"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer

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
import theano.tensor.nlinalg as Tla
import theano.tensor.slinalg as Tsla
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

class GenerativeModel(object):
    '''
    Interface class for generative time-series models
    '''
    def __init__(self,GenerativeParams,xDim,yDim,srng = None,nrng = None):

        # input variable referencing top-down or external input

        self.xDim = xDim
        self.yDim = yDim

        self.srng = srng
        self.nrng = nrng

        # internal RV for generating sample
        self.Xsamp = T.matrix('Xsamp')

    def evaluateLogDensity(self):
        '''
        Return a theano function that evaluates the density of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def getParams(self):
        '''
        Return parameters of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def generateSamples(self):
        '''
        generates joint samples
        '''
        raise Exception('Cannot call function of interface class')

    def __repr__(self):
        return "GenerativeModel"

class LDS(GenerativeModel):
    '''
    Gaussian latent LDS with (optional) NN observations:

    x(0) ~ N(x0, Q0 * Q0')
    x(t) ~ N(A x(t-1), Q * Q')
    y(t) ~ N(NN(x(t)), R * R')

    For a Kalman Filter model, choose the observation network, NN(x), to be
    a one-layer network with a linear output. The latent state has dimensionality
    n (parameter "xDim") and observations have dimensionality m (parameter "yDim").

    Inputs:
    (See GenerativeModel abstract class definition for a list of standard parameters.)

    GenerativeParams  -  Dictionary of LDS parameters
                           * A     : [n x n] linear dynamics matrix; should
                                     have eigenvalues with magnitude strictly
                                     less than 1
                           * QChol : [n x n] square root of the innovation
                                     covariance Q
                           * Q0Chol: [n x n] square root of the innitial innovation
                                     covariance
                           * RChol : [n x 1] square root of the diagonal of the
                                     observation covariance
                           * x0    : [n x 1] mean of initial latent state
                           * NN_XtoY_Params:
                                   Dictionary with one field:
                                    - network: a lasagne network with input
                                      dimensionality n and output dimensionality m
    '''
    def __init__(self, GenerativeParams, xDim, yDim, srng = None, nrng = None):

        super(LDS, self).__init__(GenerativeParams,xDim,yDim,srng,nrng)

        # parameters
        if 'A' in GenerativeParams:
            self.A      = theano.shared(value=GenerativeParams['A'].astype(theano.config.floatX), name='A'     ,borrow=True)     # dynamics matrix
        else:
            # TBD:MAKE A BETTER WAY OF SAMPLING DEFAULT A
            self.A      = theano.shared(value=.5*np.diag(np.ones(xDim).astype(theano.config.floatX)), name='A'     ,borrow=True)     # dynamics matrix

        if 'QChol' in GenerativeParams:
            self.QChol  = theano.shared(value=GenerativeParams['QChol'].astype(theano.config.floatX), name='QChol' ,borrow=True)     # cholesky of innovation cov matrix
        else:
            self.QChol  = theano.shared(value=(np.eye(xDim)).astype(theano.config.floatX), name='QChol' ,borrow=True)     # cholesky of innovation cov matrix

        if 'Q0Chol' in GenerativeParams:
            self.Q0Chol = theano.shared(value=GenerativeParams['Q0Chol'].astype(theano.config.floatX), name='Q0Chol',borrow=True)     # cholesky of starting distribution cov matrix
        else:
            self.Q0Chol = theano.shared(value=(np.eye(xDim)).astype(theano.config.floatX), name='Q0Chol',borrow=True)     # cholesky of starting distribution cov matrix

        if 'RChol' in GenerativeParams:
            self.RChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['RChol'].astype(theano.config.floatX)), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.RChol  = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX)/10, name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix

        if 'x0' in GenerativeParams:
            self.x0     = theano.shared(value=GenerativeParams['x0'].astype(theano.config.floatX), name='x0'    ,borrow=True)     # set to zero for stationary distribution
        else:
            self.x0     = theano.shared(value=np.zeros((xDim,)).astype(theano.config.floatX), name='x0'    ,borrow=True)     # set to zero for stationary distribution

        if 'NN_XtoY_Params' in GenerativeParams:
            self.NN_XtoY = GenerativeParams['NN_XtoY_Params']['network']
        else:
            # Define a neural network that maps the latent state into the output
            gen_nn = lasagne.layers.InputLayer((None, xDim))
            self.NN_XtoY = lasagne.layers.DenseLayer(gen_nn, yDim, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Orthogonal())

        # set to our lovely initial values
        if 'C' in GenerativeParams:
            self.NN_XtoY.W.set_value(GenerativeParams['C'].astype(theano.config.floatX))
        if 'd' in GenerativeParams:
            self.NN_XtoY.b.set_value(GenerativeParams['d'].astype(theano.config.floatX))

        # we assume diagonal covariance (RChol is a vector)
        self.Rinv    = 1./(self.RChol**2) #Tla.matrix_inverse(T.dot(self.RChol ,T.transpose(self.RChol)))
        self.Lambda  = Tla.matrix_inverse(T.dot(self.QChol ,self.QChol.T))
        self.Lambda0 = Tla.matrix_inverse(T.dot(self.Q0Chol,self.Q0Chol.T))

        # Call the neural network output a rate, basically to keep things consistent with the PLDS class
        self.rate = lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp)

    def sampleX(self, _N):
        _x0 = np.asarray(self.x0.eval(), dtype=theano.config.floatX)
        _Q0Chol = np.asarray(self.Q0Chol.eval(), dtype=theano.config.floatX)
        _QChol = np.asarray(self.QChol.eval(), dtype=theano.config.floatX)
        _A = np.asarray(self.A.eval(), dtype=theano.config.floatX)

        norm_samp = np.random.randn(_N, self.xDim).astype(theano.config.floatX)
        x_vals = np.zeros([_N, self.xDim]).astype(theano.config.floatX)

        x_vals[0] = _x0 + np.dot(norm_samp[0],_Q0Chol.T)

        for ii in xrange(_N-1):
            x_vals[ii+1] = x_vals[ii].dot(_A.T) + norm_samp[ii+1].dot(_QChol.T)

        return x_vals.astype(theano.config.floatX)

    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.rate+T.dot(self.srng.normal([self.Xsamp.shape[0],self.yDim]),T.diag(self.RChol).T)

    def sampleXY(self, _N):
        ''' Return numpy samples from the generative model. '''
        X = self.sampleX(_N)
        nprand = np.random.randn(X.shape[0],self.yDim).astype(theano.config.floatX)
        _RChol = np.asarray(self.RChol.eval(), dtype=theano.config.floatX)
        Y = self.rate.eval({self.Xsamp: X}) + np.dot(nprand,np.diag(_RChol).T)
        return [X,Y]

    def getParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] + [self.RChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY)

    def evaluateLogDensity(self,X,Y):
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        resX  = X[1:]-T.dot(X[:(X.shape[0]-1)],self.A.T)
        resX0 = X[0]-self.x0

        LogDensity  = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum() - (0.5*T.dot(resX.T,resX)*self.Lambda).sum() - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T)
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y.shape[0] + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0))  - 0.5*(self.xDim + self.yDim)*np.log(2*np.pi)*Y.shape[0]

        return LogDensity


class FLDS(LDS):
    '''
    x(0) ~ N(x0, Q0 * Q0')
    x(t) ~ N(A x(t-1) + B u(t-1), Q * Q')
    y(t) ~ y(t-1) + N(NN(x(t)) + D tanh(u(t-1)), R * R')
    u(t) ~ K * y(t) - convolution

    Gamma is a vector that holds a multiplier for each latent. We can penalize this to
    ensure that information isn't diluted to many latents by enabling lmda and theta.

    Constrain latents to unit variance the following term in the ELBO:
    -theta * (var(z) - 1)**2
    '''
    def __init__(self, GenerativeParams, xDim, yDim, y_extDim=None, srng = None, nrng = None):
        super(FLDS, self).__init__(GenerativeParams,xDim,yDim,srng,nrng)

        if 'y0' in GenerativeParams:
            self.y0 = theano.shared(value=GenerativeParams['y0'].astype(theano.config.floatX), name='y0', borrow=True)
        else:
            self.y0 = theano.shared(value=np.zeros((xDim + yDim,)).astype(theano.config.floatX), name='y0', borrow=True)

        if 'reg' in GenerativeParams:
            self.reg = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['reg']), name='reg', borrow=True)
        else:
            self.reg = None


        self.gamma = theano.shared(value=np.ones((xDim)).astype(theano.config.floatX), name='gamma', borrow=True)
        if 'lmda' in GenerativeParams:
            self.lmda = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['lmda']), name='lmda', borrow=True)
        else:
            self.lmda = None

        if 'theta' in GenerativeParams:
            self.theta = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['theta']), name='theta', borrow=True)
        else:
            self.theta = None

        if 'filter_size' in GenerativeParams:
            filter_size = GenerativeParams['filter_size']
        else:
            filter_size = 5

        if 'B' in GenerativeParams:
            self.B = theano.shared(value=GenerativeParams['B'].astype(theano.config.floatX), name='B', borrow=True)
        else:
            self.B = theano.shared(value=np.zeros((yDim, xDim)).astype(theano.config.floatX), name='B', borrow=True)

        if 'D' in GenerativeParams:
            self.D = theano.shared(value=GenerativeParams['D'].astype(theano.config.floatX), name='D', borrow=True)
        else:
            self.D = theano.shared(value=np.zeros((yDim, yDim)).astype(theano.config.floatX), name='D', borrow=True)

        if 'CNN_YtoU_Params' in GenerativeParams:
            self.CNN_YtoU = GenerativeParams['CNN_YtoU_Params']['network']
        else:
            gen_nn = lasagne.layers.InputLayer((yDim, None))
            gen_nn = lasagne.layers.ReshapeLayer(gen_nn, (1, 1, yDim, [1]))
            gen_nn = lasagne.layers.Conv2DLayer(gen_nn, yDim, (yDim, filter_size),
                                                nonlinearity=lasagne.nonlinearities.linear,
                                                pad=(0, filter_size))
            self.CNN_YtoU = lasagne.layers.ReshapeLayer(gen_nn, (yDim, -1))

        self.Y = T.matrix('Y')
        self.u = lasagne.layers.get_output(self.CNN_YtoU, inputs=self.Y).T[:-(filter_size+1)]

        if y_extDim is not None:
            if 'CNN_YexttoU_Params' in GenerativeParams:
                self.CNN_YexttoU = GenerativeParams['CNN_YexttoU_Params']['network']
            else:
                gen_nn = lasagne.layers.InputLayer((y_extDim, None))
                gen_nn = lasagne.layers.ReshapeLayer(gen_nn, (1, 1, y_extDim, [1]))
                gen_nn = lasagne.layers.Conv2DLayer(gen_nn, yDim, (y_extDim, filter_size),
                                                    nonlinearity=lasagne.nonlinearities.linear,
                                                    pad=(0, filter_size))
                self.CNN_YexttoU = lasagne.layers.ReshapeLayer(gen_nn, (yDim, -1))
            self.Y_ext = T.matrix('Y_ext')
            self.u += lasagne.layers.get_output(self.CNN_YexttoU, inputs=self.Y_ext).T[:-(filter_size+1)]
        else:
            self.Y_ext = None

        # add previous position so that we're modeling velecity rather than position
        self.rate = T.horizontal_stack(self.Y[:, (0,)], self.Y[:, :-1]).T
        self.rate += lasagne.layers.get_output(self.NN_XtoY, inputs=self.Xsamp)
        self.rate += T.dot(T.tanh(self.u), self.D)
        self.latent_rate = T.dot(self.u, self.B)

    def sampleX(self, _N):
        # how do you sample X when it relies on observations? Evaluate rate and feed it back in?
        raise NotImplementedError()

    def getParams(self):
        rets = [self.A] + [self.QChol] + [self.Q0Chol] + [self.RChol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY)
        rets += lasagne.layers.get_all_params(self.CNN_YtoU) + [self.B] + [self.D]
        if self.Y_ext is not None:
            rets += lasagne.layers.get_all_params(self.CNN_YexttoU)
        if self.lmda is not None and self.theta is not None:
            rets += [self.gamma]
        return rets

    def evaluateLogDensity(self, X, Y, Y_ext=None):
        '''
        Ignores first observation in Y since you need a previous obs (should this be changed later?)
        '''
        X = X * self.gamma

        if self.Y_ext is not None:
            Ypred = theano.clone(self.rate, replace={self.Xsamp: X,
                                                     self.Y: Y.T,
                                                     self.Y_ext: Y_ext.T})
        else:
            Ypred = theano.clone(self.rate, replace={self.Xsamp: X,
                                                     self.Y: Y.T})
        resY = Y - Ypred

        curr_X = X[1:, :]
        Xpred = T.dot(X[:-1], self.A)
        if self.Y_ext is not None:
            Xpred += theano.clone(self.latent_rate, replace={self.Y: Y.T,
                                                             self.Y_ext: Y_ext.T})[1:]
        else:
            Xpred += theano.clone(self.latent_rate, replace={self.Y: Y.T})[1:]
        resX = curr_X - Xpred

        resX0 = X[0] - self.x0

        LogDensity = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum() - (0.5*T.dot(resX.T,resX)*self.Lambda).sum() - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T)
        if self.reg is not None:
            K, _ = lasagne.layers.get_all_params(self.CNN_YtoU)
            Kp, _ = lasagne.layers.get_all_params(self.CNN_YexttoU)
            LogDensity -= self.reg * (T.abs_(K).sum() + T.abs_(Kp).sum() + T.abs_(self.D).sum() + T.abs_(self.B).sum())  # add regularization to filters and mixing matrices
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y.shape[0] + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0))  - 0.5*(self.xDim + self.yDim)*np.log(2*np.pi)*Y.shape[0]

        if self.lmda is not None and self.theta is not None:
            LogDensity -= self.lmda * np.abs(self.gamma).sum()
            LogDensity -= self.theta * ((X.var(axis=0) - 1)**2).sum()

        return LogDensity


class SFLDS():
    '''
    y(t) ~ y(t-1) + N(D tanh(NN(y(t-1), u(t-1))), R * R')
    u(t) ~ K y(t) - convolution
    '''
    def __init__(self, GenerativeParams, yDim, y_extDim=None, srng=None, nrng=None):
        self.yDim = yDim
        self.srng = srng
        self.nrng = nrng

        if 'RChol' in GenerativeParams:
            self.RChol  = theano.shared(value=np.ndarray.flatten(GenerativeParams['RChol'].astype(theano.config.floatX)), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.RChol  = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX)/10, name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix

        if 'y0' in GenerativeParams:
            self.y0 = theano.shared(value=GenerativeParams['y0'].astype(theano.config.floatX), name='y0', borrow=True)
        else:
            self.y0 = theano.shared(value=np.zeros((yDim,)).astype(theano.config.floatX), name='y0', borrow=True)

        if 'reg' in GenerativeParams:
            self.reg = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['reg']), name='reg', borrow=True)
        else:
            self.reg = None

        if 'D' in GenerativeParams:
            self.D = theano.shared(value=GenerativeParams['D'].astype(theano.config.floatX), name='D', borrow=True)
        else:
            self.D = theano.shared(value=np.zeros((yDim, yDim)).astype(theano.config.floatX), name='D', borrow=True)

        if 'NN_YUprevtoY_Params' in GenerativeParams:
            # a neural network that maps the observations and control input to next observation
            # this network must have a tanh nonlinearity on the last layer
            self.NN_YUprevtoY = GenerativeParams['NN_YUprevtoY_Params']['networks']
            self.NN_layers = GenerativeParams['NN_YUprevtoY_Params']['num_layers']
        else:
            # default is to simply add a nonlinearity with no neural net.
            self.NN_layers = 0

        if 'filter_size' in GenerativeParams:
            self.filter_size = GenerativeParams['filter_size']
        else:
            self.filter_size = 5

        self.Rinv = 1./(self.RChol**2) #Tla.matrix_inverse(T.dot(self.RChol ,T.transpose(self.RChol)))

        if 'CNN_YtoU_Params' in GenerativeParams:
            self.CNN_YtoU = GenerativeParams['CNN_YtoU_Params']['network']
        else:
            gen_nn = lasagne.layers.InputLayer((yDim, None))
            gen_nn = lasagne.layers.ReshapeLayer(gen_nn, (1, 1, yDim, [1]))
            gen_nn = lasagne.layers.Conv2DLayer(gen_nn, yDim, (yDim, self.filter_size),
                                                nonlinearity=lasagne.nonlinearities.linear,
                                                pad=(0, self.filter_size))
            self.CNN_YtoU = lasagne.layers.ReshapeLayer(gen_nn, (yDim, -1))

        self.yDim = yDim
        self.Y = T.matrix('Y')
        self.u = lasagne.layers.get_output(self.CNN_YtoU, inputs=self.Y).T[:-(self.filter_size+1)]

        if y_extDim is not None:
            if 'CNN_YexttoU_Params' in GenerativeParams:
                self.CNN_YexttoU = GenerativeParams['CNN_YexttoU_Params']['network']
            else:
                gen_nn = lasagne.layers.InputLayer((y_extDim, None))
                gen_nn = lasagne.layers.ReshapeLayer(gen_nn, (1, 1, y_extDim, [1]))
                gen_nn = lasagne.layers.Conv2DLayer(gen_nn, yDim, (y_extDim, self.filter_size),
                                                    nonlinearity=lasagne.nonlinearities.linear,
                                                    pad=(0, self.filter_size))
                self.CNN_YexttoU = lasagne.layers.ReshapeLayer(gen_nn, (yDim, -1))
            self.Y_ext = T.matrix('Y_ext')
            self.u += lasagne.layers.get_output(self.CNN_YexttoU, inputs=self.Y_ext).T[:-(self.filter_size+1)]
        else:
            self.Y_ext = None

        self.rate = T.horizontal_stack(self.Y[:, (0,)], self.Y[:, :-1]).T
        Yprev = T.horizontal_stack(self.Y[:, (0,)], self.Y[:, :-1]).T
        if self.NN_layers == 0:
            control_out = T.tanh(self.u)
        else:
            if self.Y_ext is not None:
                Yextprev = T.horizontal_stack(self.Y_ext[:, (0,)], self.Y_ext[:, :-1]).T
                YUprev = T.horizontal_stack(Yprev, Yextprev, self.u)
            else:
                YUprev = T.horizontal_stack(Yprev, self.u)
            control_out = []
            for i in range(self.yDim):
                control_out.append(lasagne.layers.get_output(self.NN_YUprevtoY[i], inputs=YUprev))
            if len(control_out) >= 2:
                control_out = T.concatenate(control_out, axis=1)
            else:
                control_out = control_out[0]

        self.rate = Yprev + T.dot(control_out, self.D)

    def getParams(self):
        rets = [self.D] + [self.RChol] + lasagne.layers.get_all_params(self.CNN_YtoU)
        if self.Y_ext is not None:
            rets += lasagne.layers.get_all_params(self.CNN_YexttoU)
        if self.NN_layers > 0:
            for nn in self.NN_YUprevtoY:
                rets += lasagne.layers.get_all_params(nn)
        return rets

    def getNextState(self, curr_y, curr_y_ext):
        pred_u = lasagne.layers.get_output(self.CNN_YtoU,
                                           inputs=curr_y.T).T[-(self.filter_size+1):-(self.filter_size)]
        pred_u += lasagne.layers.get_output(self.CNN_YexttoU,
                                            inputs=curr_y_ext.T).T[-(self.filter_size+1):-(self.filter_size)]
        
        pred_y = curr_y[-1]
        if self.NN_layers == 0:
            control_out = T.tanh(pred_u)
        else:
            if self.Y_ext is not None:
                YUprev = T.concatenate((pred_y.reshape((1, -1)),
                                        curr_y_ext[-1].reshape((1, -1)),
                                        pred_u), axis=1)
            else:
                YUprev = T.concatenate((pred_y.reshape((1, -1)),
                                        pred_u), axis=1)
            control_out = []
            for i in range(self.yDim):
                control_out.append(lasagne.layers.get_output(self.NN_YUprevtoY[i], inputs=YUprev))
            if len(control_out) >= 2:
                control_out = T.concatenate(control_out, axis=1)
            else:
                control_out = control_out[0]
        pred_y += T.dot(control_out, self.D)
        pred_y += T.dot(self.srng.normal((self.yDim,)),
                        T.diag(self.RChol).T)

        return pred_y

    def evaluateLogDensity(self, Y, Y_ext=None):
        '''
        Ignores first observation in Y since you need a previous obs (should this be changed later?)
        '''
        if self.Y_ext is not None:
            Ypred = theano.clone(self.rate, replace={self.Y: Y.T,
                                                     self.Y_ext: Y_ext.T})
        else:
            Ypred = theano.clone(self.rate, replace={self.Y: Y.T})
        resY = Y - Ypred

        LogDensity = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum()
        if self.reg is not None:
            K, _ = lasagne.layers.get_all_params(self.CNN_YtoU)
            Kp, _ = lasagne.layers.get_all_params(self.CNN_YexttoU)
            LogDensity -= self.reg * (T.abs_(K).sum() + T.abs_(Kp).sum() + T.abs_(self.D).sum())  # add regularization to filters
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y.shape[0] - 0.5*(self.yDim)*np.log(2*np.pi)*Y.shape[0]

        return LogDensity


class GPLDS():
    def __init__(self, GenerativeParams, yDim, ntrials, srng=None, nrng=None):
        self.yDim = yDim
        self.ntrials = ntrials

        self.srng = srng
        self.nrng = nrng

        if 'RChol' in GenerativeParams:
            self.RChol = theano.shared(value=np.ndarray.flatten(GenerativeParams['RChol'].astype(theano.config.floatX)), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.RChol = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX)/10, name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix

        # size of time domain filter
        if 'filter_size' in GenerativeParams:
            self.filter_size = theano.shared(value=GenerativeParams['filter_size'], name='filter_size',
                                             borrow=True)
        else:
            self.filter_size = theano.shared(value=5, name='filter_size',
                                             borrow=True)
        # space between bins in x direction of ball
        if 'dx' in GenerativeParams:
            self.dx = GenerativeParams['dx']
        else:
            self.dx = 0.05
        # a, b, c, and d are unconstrained scalars that will form our contrained
        # S matrix
        if 'a' in GenerativeParams:
            self.a = theano.shared(value=GenerativeParams['a'], name='a',
                                   borrow=True)
        else:
            self.a = theano.shared(value=1.0, name='a',
                                   borrow=True)
        if 'b' in GenerativeParams:
            self.b = theano.shared(value=GenerativeParams['b'], name='b',
                                   borrow=True)
        else:
            self.b = theano.shared(value=0.0, name='b',
                                   borrow=True)
        if 'c' in GenerativeParams:
            self.c = theano.shared(value=GenerativeParams['c'], name='c',
                                   borrow=True)
        else:
            self.c = theano.shared(value=1.0, name='c',
                                   borrow=True)
        if 'd' in GenerativeParams:
            self.d = theano.shared(value=GenerativeParams['d'], name='d',
                                   borrow=True)
        else:
            self.d = theano.shared(value=0.0, name='d',
                                   borrow=True)

        if 'p' in GenerativeParams:
            self.p = GenerativeParams['p']
        else:
            self.p = 10.0
        if 'q' in GenerativeParams:
            self.q = GenerativeParams['q']
        else:
            self.q = 10.0
        # learnable velocity for each observation dimension
        if 'vel' in GenerativeParams:
            self.vel = theano.shared(value=GenerativeParams['vel'], name='vel',
                                     borrow=True)
        else:
            self.vel = theano.shared(value=np.ones(self.yDim), name='vel',
                                     borrow=True)

        # create constrained parameters from uncontrained a, b, c, d
        self.theta = T.exp(self.a)
        self.phi = (self.theta / 2) * T.cos(np.pi / (self.p + 1)) * T.nnet.sigmoid(self.b)
        self.omega = T.exp(self.c)
        self.tau = (self.omega / 2) * T.cos(np.pi / (self.q + 1)) * T.nnet.sigmoid(self.d)
        lambda_x = T.stack((self.phi, self.theta, self.phi), axis=0)
        lambda_t = T.stack((self.tau, self.omega, self.tau), axis=0)
        self.S = T.outer(lambda_x, lambda_t)

        self.Rinv = 1. / (self.RChol**2)  #Tla.matrix_inverse(T.dot(self.RChol ,T.transpose(self.RChol)))
        n_xbins = int(1 / self.dx) + 1
        self.K = theano.shared(value=0.05 * np.random.randn(yDim, yDim,
                                                            n_xbins, self.filter_size.eval())
                               .astype(theano.config.floatX),
                               name='K', borrow=True)


    def interpolate_filters(self, data):
        """
        Returns interpolated filters (based on x-position) for each time step
        """
        # get max possible grid location
        max_grid = int(1.0 / self.dx) + 1
        # get index of grid location
        gridloc = data[:, 1] / self.dx
        # constrain to within grid
        gridloc = T.maximum(gridloc, 0)
        gridloc = T.minimum(gridloc, max_grid)
        # how much to use of the lower bound filter
        mix = (gridloc - T.floor(gridloc)).reshape((1, 1, -1, 1))
        # interpolate filters for each data point
        d_filts = (mix * self.K[:, :, T.floor(gridloc).astype('int16'), :] +
                   (1 - mix) * self.K[:, :, T.ceil(gridloc).astype('int16'), :])

        return d_filts

    def split_data(self, data, col):
        """
        Splits data into a row for each step in a convolution. Similar to im2col
        """
        out, _ = theano.scan(fn=lambda idx, vect, length: vect[idx:idx + length],
                             sequences=[T.arange(data.shape[0] -
                                                 self.filter_size + 1)],
                             non_sequences=[data[:, col], self.filter_size])
        return out

    def getNextState(self, curr_y):
        """
        Return predicted next data point based on given point
        """
        filt = self.interpolate_filters(curr_y)[:, :, -1, :]
        Y_pred = []
        for i in range(self.yDim):  # to
            control = (filt[:, i, :].T * curr_y).sum()
            Y_pred.append(control)
        Y_pred = T.stack(Y_pred, axis=0)
        Y_pred = T.tanh(Y_pred)
        Y_pred *= self.vel
        Y_pred += curr_y[-1]  # previous state
        Y_pred += T.dot(self.srng.normal((self.yDim,)),
                        T.diag(self.RChol).T)  # noise
        return Y_pred

    def fit_trial(self, Y_true):
        '''
        Return a theano function that calculates a fit for the given data.
        '''
        filts = self.interpolate_filters(Y_true)[:, :, :-1, :]
        pad = np.zeros((self.filter_size.eval() - 1, self.yDim))
        pad[:, 1] = 0.2
        pad = theano.shared(value=pad)
        Y_pad = T.vertical_stack(pad, Y_true[:-1, :])
        Y_pred = []
        for j in range(self.yDim):  # to
            Y_pred.append(T.zeros((Y_true.shape[0] - 1, 1)))
            for i in range(self.yDim):  # from
                split_in = self.split_data(Y_pad, i)
                Y_pred[j] += self.vel[j] * T.tanh((split_in * filts[i, j, :, :])
                                                  .sum(axis=1, keepdims=True))
        Y_pred = T.horizontal_stack(*Y_pred)
        Y_pred += Y_true[:-1]  # add previous observation
        return Y_pred

    def evaluateLogDensity(self, Y_true):
        '''
        Return a theano function that evaluates the density of the GenerativeModel.
        '''
        # don't need last filter since it will be used to predict the observation
        # that succeeds our current data.
        filts = self.interpolate_filters(Y_true)[:, :, :-1, :]
        pad = np.zeros((self.filter_size.eval() - 1, self.yDim))
        pad[:, 1] = 0.2
        pad = theano.shared(value=pad)
        Y_pad = T.vertical_stack(pad, Y_true[:-1, :])
        Y_pred = []
        for j in range(self.yDim):  # to
            Y_pred.append(T.zeros((Y_true.shape[0] - 1, 1)))
            for i in range(self.yDim):  # from
                split_in = self.split_data(Y_pad, i)
                Y_pred[j] += self.vel[j] * T.tanh((split_in * filts[i, j, :, :])
                                                  .sum(axis=1, keepdims=True))
        Y_pred = T.horizontal_stack(*Y_pred)
        Y_pred += Y_true[:-1]  # add previous observation

        resY = Y_true[1:] - Y_pred  # get errors for each point

        LogDensity = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum()
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y_true.shape[0] - 0.5*(self.yDim)*np.log(2*np.pi)*Y_true.shape[0]

        def sym_tridiag_det(a, b, length):
            """
            Computes the determinant of a symmetric tridiagonal
            matrix with repeating diagonals

            https://en.wikipedia.org/wiki/Tridiagonal_matrix#Determinant

            a (float): value of center diagonal
            b (float): value of up-1 and down-1 diagonals
            length (int): length of center diagonal
            """
            f_lag = 0.0
            f = 1.0
            for i in xrange(length):
                f_next = a * f - (b**2) * f_lag
                f_lag = f
                f = f_next
            return f

        # iterate over all filters
        for i in range(self.yDim):
            for j in range(self.yDim):
                curr_filt = self.K[i, j, :, :]
                conv_res = T.signal.conv.conv2d(curr_filt, self.S[::-1, ::-1],
                                                border_mode='full')[1:-1, 1:-1]
                LogDensity -= (0.5 * (curr_filt * conv_res).sum()) / self.ntrials

        det_x = sym_tridiag_det(self.theta, self.phi,
                                int(self.p))
        det_t = sym_tridiag_det(self.omega, self.tau,
                                int(self.q))
        LogDensity += ((self.q / 2.0) * T.log(T.abs_(det_x))) * (self.yDim**2 / float(self.ntrials))
        LogDensity += ((self.p / 2.0) * T.log(T.abs_(det_t))) * (self.yDim**2 / float(self.ntrials))

        return LogDensity

    def getParams(self):
        '''
        Return parameters of the GenerativeModel.
        '''
        rets = [self.K] + [self.a] + [self.b] + [self.c] + [self.d]
        rets += [self.vel] + [self.RChol]
        return rets

    def __repr__(self):
        return "GenerativeGPModel"


class GPLDS2(GenerativeModel):
    """
    Same as GPLDS but with optional latents (separate for ball and goalie).
    All variables pertaining to latents (A, QChol, Q0Chol, Lambda, etc.) are
    2 element tuples w/ the first element being for the goalie and second for
    the shooter.
    """
    def __init__(self, GenerativeParams, xDim, yDim, ntrials, srng=None, nrng=None, noX=False):
        super(GPLDS2, self).__init__(GenerativeParams, xDim, yDim, srng, nrng)
        self.noX = noX
        self.ntrials = ntrials

        # dynamics matrix
        self.A = (theano.shared(value=.5*np.diag(np.ones(xDim[0]).astype(theano.config.floatX)), name='A_g', borrow=True),
                  theano.shared(value=.5*np.diag(np.ones(xDim[1]).astype(theano.config.floatX)), name='A_b', borrow=True))
        # cholesky of innovation cov matrix
        self.QChol = (theano.shared(value=(np.eye(xDim[0])).astype(theano.config.floatX), name='QChol_g', borrow=True),
                      theano.shared(value=(np.eye(xDim[1])).astype(theano.config.floatX), name='QChol_b', borrow=True))
        # cholesky of starting distribution cov matrix
        self.Q0Chol = (theano.shared(value=(np.eye(xDim[0])).astype(theano.config.floatX), name='Q0Chol_g',borrow=True),
                       theano.shared(value=(np.eye(xDim[1])).astype(theano.config.floatX), name='Q0Chol_b',borrow=True))
        # set to zero for stationary distribution
        self.x0 = (theano.shared(value=np.zeros((xDim[0],)).astype(theano.config.floatX), name='x0_g', borrow=True),
                   theano.shared(value=np.zeros((xDim[1],)).astype(theano.config.floatX), name='x0_b', borrow=True))
        # cholesky of observation noise cov matrix
        if 'RChol' in GenerativeParams:
            self.RChol = theano.shared(value=GenerativeParams['RChol'].astype(theano.config.floatX), name='RChol', borrow=True)
        else:
            self.RChol = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX) / 10, name='RChol', borrow=True)

        # we assume diagonal covariance (RChol is a vector)
        self.Rinv = 1. / (self.RChol**2)  #Tla.matrix_inverse(T.dot(self.RChol ,T.transpose(self.RChol)))
        invert = lambda mat: Tla.matrix_inverse(T.dot(mat, mat.T))
        self.Lambda = map(invert, self.QChol)
        self.Lambda0 = map(invert, self.Q0Chol)

        # size of time domain filter
        if 'err_filter_size' in GenerativeParams:
            self.err_filter_size = theano.shared(value=GenerativeParams['err_filter_size'], name='err_filter_size',
                                                 borrow=True)
        else:
            self.err_filter_size = theano.shared(value=5, name='err_filter_size',
                                                 borrow=True)

        if 'ext_filter_size' in GenerativeParams:
            self.ext_filter_size = theano.shared(value=GenerativeParams['ext_filter_size'], name='ext_filter_size',
                                                 borrow=True)
        else:
            self.ext_filter_size = theano.shared(value=5, name='ext_filter_size',
                                                 borrow=True)

        # a, b, c, and d are unconstrained scalars that will form our contrained
        # S matrix
        if 'a' in GenerativeParams:
            self.a = theano.shared(value=GenerativeParams['a'], name='a',
                                   borrow=True)
        else:
            self.a = theano.shared(value=1.0, name='a',
                                   borrow=True)
        if 'b' in GenerativeParams:
            self.b = theano.shared(value=GenerativeParams['b'], name='b',
                                   borrow=True)
        else:
            self.b = theano.shared(value=0.0, name='b',
                                   borrow=True)

        if 'p' in GenerativeParams:
            self.p = GenerativeParams['p']
        else:
            self.p = 10.0
        # learnable velocity for each observation dimension
        if 'log_vel' in GenerativeParams:
            self.log_vel = theano.shared(value=GenerativeParams['log_vel'], name='log_vel',
                                     borrow=True)
        else:
            self.log_vel = theano.shared(value=np.ones(self.yDim), name='log_vel',
                                     borrow=True)
        self.vel = T.exp(self.log_vel)
        if 'yCols' in GenerativeParams:
            self.yCols = GenerativeParams['yCols']
        else:
            self.yCols = [0, 1, 1]

        # Number of time steps forward to make prediction in loss function
        # Using a large number here forces predictions to be more accurate
        # than simply estimating the current velocity and using that as the
        # next velocity
        if 'forward' in GenerativeParams:
            self.forward = GenerativeParams['forward']
        else:
            self.forward = 1

        if 'lag' in GenerativeParams:
            self.lag = GenerativeParams['lag']
        else:
            self.lag = 1

        # create constrained parameters from uncontrained a, b
        self.theta = T.exp(self.a)
        self.phi = (self.theta / 2) * T.cos(np.pi / (self.p + 1)) * T.nnet.sigmoid(self.b)
        self.S = T.stack((self.phi, self.theta, self.phi), axis=0)

        self.c = {}  # unconstrained diagonals
        self.d = {}  # unconstrained lower triangle
        self.L = {}  # combined contrained L matrix
        self.K_mu = {}  # mean filters
        self.K_b = {}  # bias for each filter
        for i in range(self.yDim):
            for j in range(self.yDim):
                if i == j:
                    filter_size = self.err_filter_size.eval()
                elif self.yCols[i] != self.yCols[j]:
                    filter_size = self.ext_filter_size.eval()
                else:
                    continue
                self.K_mu[(i, j)] = theano.shared(value=0.05 * np.random.randn(filter_size)
                                                  .astype(theano.config.floatX),
                                                  name='K_mu_%ito%i' % (i, j), borrow=True)
                self.K_b[(i, j)] = theano.shared(value=0.0,
                                                 name='K_b_%ito%i' % (i, j), borrow=True)
                self.c[(i, j)] = theano.shared(value=(-2 * np.ones(filter_size))
                                               .astype(theano.config.floatX),
                                               name='c_%ito%i' % (i, j), borrow=True)
                self.d[(i, j)] = theano.shared(value=np.zeros((filter_size, filter_size))
                                               .astype(theano.config.floatX),
                                               name='d_%ito%i' % (i, j), borrow=True)
                Ldiag = T.exp(self.c[(i, j)])  # diagonal must be positive
                self.L[(i, j)] = T.tril(self.d[(i, j)], k=-1) + T.diag(Ldiag)

    def sample_K(self):
        """
        Obtain a sample filter, K, from the gaussian distribution
        """
        K = {}
        for i in range(self.yDim):
            for j in range(self.yDim):
                if i == j:
                    K[(i, j)] = (self.K_mu[(i, j)] +
                                 self.L[(i, j)].dot(
                                 self.srng.normal((self.err_filter_size,))))
                elif self.yCols[i] != self.yCols[j]:
                    K[(i, j)] = (self.K_mu[(i, j)] +
                                 self.L[(i, j)].dot(
                                 self.srng.normal((self.ext_filter_size,))))
        return K

    def evalEntropy(self):
        """
        Return the entropy term for K in the ELBO
        """
        entropy = 0
        for i in range(self.yDim):
            for j in range(self.yDim):
                if i == j:
                    Ldiag = T.diag(self.L[(i, j)])
                    entropy += (self.err_filter_size / 2.0) * (1 + T.log(2 * np.pi))
                    entropy += T.log(Ldiag).sum()
                elif self.yCols[i] != self.yCols[j]:
                    Ldiag = T.diag(self.L[(i, j)])
                    entropy += (self.ext_filter_size / 2.0) * (1 + T.log(2 * np.pi))
                    entropy += T.log(Ldiag).sum()
        entropy /= float(self.ntrials)
        return entropy

    def split_data(self, data, col, ext=False):
        """
        Splits data into a row for each step in a convolution. Similar to im2col
        """
        if ext:
            out, _ = theano.scan(fn=lambda idx, vect, length: vect[idx:idx + length],
                                 sequences=[T.arange(data.shape[0] -
                                                     self.ext_filter_size + 1)],
                                 non_sequences=[data[:, col], self.ext_filter_size])
        else:
            out, _ = theano.scan(fn=lambda idx, vect, length: vect[idx:idx + length],
                                 sequences=[T.arange(data.shape[0] -
                                                     self.err_filter_size + 1)],
                                 non_sequences=[data[:, col], self.err_filter_size])
        return out

    def getNextState(self, K, curr_xg, curr_xb, curr_y):
        """
        Return predicted next data point based on given point
        """
        if not self.noX:
            curr_x = T.horizontal_stack(curr_xg, curr_xb)
        else:
            curr_x = None

        Ypred = self.get_Ypred(curr_x, curr_y, K)[-1]
        Ypred += T.dot(self.srng.normal((self.yDim,)),
                       T.diag(self.RChol).T)  # noise
        return Ypred

    def get_Ypred(self, X, Y_true, K, pad=None):
        if pad is not None:
            Y_pad = T.vertical_stack(pad, Y_true)
        else:
            Y_pad = Y_true.copy()
            Y_true = Y_true[self.ext_filter_size - 1 + self.lag:]
            if X is not None:
                X = X[self.ext_filter_size - 1 + self.lag:]
        error = []
        for j in range(self.yDim):  # to
            error.append(T.zeros((Y_true.shape[0],)))
            for i in range(self.yDim):  # from
                if self.yCols[i] != self.yCols[j]:
                    split_in = self.split_data(Y_pad[:-self.lag], i, ext=True)
                    error[j] += (split_in * K[(i, j)]).sum(axis=1)  # add external dynamics
        error = T.stack(error, axis=1)
        if X is not None:
            error += X  # add internal dynamics
        error -= Y_true  # subtract position to get error (setpoint - position)
        pad = T.zeros((self.err_filter_size - 1, self.yDim))
        error_pad = T.vertical_stack(pad, error)
        Ypred = []
        for i in range(self.yDim):
            split_in = self.split_data(error_pad, i)
            Ypred.append((split_in * K[(i, i)]).sum(axis=1))  # smooth errors
        Ypred = T.stack(Ypred, axis=1)
        Ypred = T.tanh(Ypred)
        Ypred = self.vel.reshape((1, self.yDim)) * Ypred
        Ypred += Y_true
        return Ypred

    def fit_trial(self, Xg, Xb, Y_true):
        '''
        Return a theano function that calculates a fit for the given data.
        '''
        K = self.sample_K()
        if not self.noX:
            X = T.horizontal_stack(Xg, Xb)
        else:
            X = None

        pad = np.zeros((self.ext_filter_size.eval() - 1 + self.lag, self.yDim))
        pad[:, 1] = 0.2
        pad = theano.shared(value=pad)
        Ypred = self.get_Ypred(X, Y_true, K, pad=pad)[:-1]
        return Ypred

    def evaluateLogDensity(self, Xg, Xb, Y_true):
        '''
        Return a theano function that evaluates the density of the GenerativeModel.
        '''
        K = self.sample_K()
        if not self.noX:
            X = T.horizontal_stack(Xg, Xb)
        else:
            X = None

        curr_Y = Y_true.copy()
        for i in range(self.forward):
            pad = [curr_Y[0, :].copy()] * (self.ext_filter_size.eval() - 1 + self.lag)
            pad = T.stack(pad, axis=0)

            Ypred = self.get_Ypred(X, curr_Y, K, pad=pad)[:-1]
            resY = Y_true[i + 1:] - Ypred  # get errors for each residual

            LogDensity = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum()
            LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y_true.shape[0] - 0.5*(self.yDim)*np.log(2*np.pi)*Y_true.shape[0]

            curr_Y = Ypred

        def get_X_LogDensity(X, i):
            curr_X = X[1:, :]
            Xpred = T.dot(X[:-1], self.A[i])
            resX = curr_X - Xpred
            resX0 = X[0] - self.x0[i]
            LD = (-(0.5 * T.dot(resX.T, resX) * self.Lambda[i]).sum() -
                  0.5 * T.dot(T.dot(resX0, self.Lambda0[i]), resX0.T))
            return LD
        if X is not None:
            LogDensity += get_X_LogDensity(Xg, 0) + get_X_LogDensity(Xb, 1)

        def sym_tridiag_det(a, b, length):
            """
            Computes the determinant of a symmetric tridiagonal
            matrix with repeating diagonals

            https://en.wikipedia.org/wiki/Tridiagonal_matrix#Determinant

            a (float): value of center diagonal
            b (float): value of up-1 and down-1 diagonals
            length (int): length of center diagonal
            """
            f_lag = 0.0
            f = 1.0
            for i in xrange(length):
                f_next = a * f - (b**2) * f_lag
                f_lag = f
                f = f_next
            return f

        # iterate over all filters
        for i in range(self.yDim):
            for j in range(self.yDim):
                if i == j or self.yCols[i] != self.yCols[j]:
                    # subtract bias bc filters don't need to be mean zero
                    curr_filt = K[(i, j)].reshape((1, -1)) - self.K_b[(i, j)]
                    conv_res = T.signal.conv.conv2d(curr_filt, self.S.reshape((1, -1)),
                                                    border_mode='full')[:, 1:-1]
                    LogDensity -= (0.5 * (curr_filt * conv_res).sum()) / self.ntrials

                    det_t = sym_tridiag_det(self.theta, self.phi,
                                            K[(i, j)].shape[0].eval())
                    LogDensity += (0.5 * T.log(T.abs_(det_t))) * (self.yDim**2 / float(self.ntrials))

        return LogDensity

    def getParams(self):
        '''
        Return parameters of the GenerativeModel.
        '''
        rets = self.K_mu.values() + [self.a] + [self.b] + self.c.values()
        rets += self.d.values() + self.K_b.values() + [self.log_vel] + [self.RChol]
        if not self.noX:
            rets += list(self.A) + list(self.QChol) + list(self.Q0Chol)
            rets += list(self.x0)
        return rets


class NLDS(GenerativeModel):
    """
    Non-Linear Dynamical System
    """
    def __init__(self, GenerativeParams, xDim, yDim, srng=None, nrng=None):
        super(NLDS, self).__init__(GenerativeParams, xDim, yDim, srng, nrng)

        self.ntrials = np.cast[theano.config.floatX](GenerativeParams['ntrials'])
        self.filter_size = GenerativeParams['filter_size']
        # dynamics matrix
        self.A = theano.shared(value=np.diag(np.ones(xDim).astype(theano.config.floatX)), name='A', borrow=True)
        # cholesky of innovation cov matrix
        self.QChol_diag = theano.shared(value=(np.ones(xDim)).astype(theano.config.floatX), name='QChol_diag', borrow=True)
        self.QChol = T.diag(self.QChol_diag)
        # cholesky of starting distribution cov matrix
        self.Q0Chol_diag = theano.shared(value=(np.ones(xDim)).astype(theano.config.floatX), name='Q0Chol_diag', borrow=True)
        self.Q0Chol = T.diag(self.Q0Chol_diag)
        # set to zero for stationary distribution
        self.x0 = theano.shared(value=np.zeros((xDim,)).astype(theano.config.floatX), name='x0', borrow=True)
        # cholesky of observation noise cov matrix
        if 'RChol' in GenerativeParams:
            self.RChol = theano.shared(value=GenerativeParams['RChol'].astype(theano.config.floatX), name='RChol', borrow=True)
        else:
            self.RChol = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX) / 10, name='RChol', borrow=True)

        # we assume diagonal covariance (RChol is a vector)
        self.Rinv = 1. / (self.RChol**2)  #Tla.matrix_inverse(T.dot(self.RChol ,T.transpose(self.RChol)))
        invert = lambda mat: Tla.matrix_inverse(T.dot(mat, mat.T))
        self.Lambda = invert(self.QChol)
        self.Lambda0 = invert(self.Q0Chol)

        # learnable velocity for each observation dimension
        if 'log_vel' in GenerativeParams:
            self.log_vel = theano.shared(value=GenerativeParams['log_vel'].astype(theano.config.floatX),
                                         name='log_vel', borrow=True)
        else:
            self.log_vel = theano.shared(value=np.ones(self.yDim).astype(theano.config.floatX),
                                         name='log_vel', borrow=True)
        self.vel = T.exp(self.log_vel)

        if 'yCols' in GenerativeParams:
            self.yCols = GenerativeParams['yCols']
        else:
            self.yCols = [0, 1, 1]

        # noise penalty constant
        if 'p' in GenerativeParams:
            self.p = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['p']), name='p', borrow=True)
        else:
            self.p = theano.shared(value=np.cast[theano.config.floatX](1000), name='p', borrow=True)

        # a, b, c, and d are unconstrained scalars that will form our contrained
        # S matrix
        if 'a' in GenerativeParams:
            self.a = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['a']), name='a',
                                   borrow=True)
        else:
            self.a = theano.shared(value=np.cast[theano.config.floatX](1.0), name='a',
                                   borrow=True)
        if 'b' in GenerativeParams:
            self.b = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['b']), name='b',
                                   borrow=True)
        else:
            self.b = theano.shared(value=np.cast[theano.config.floatX](0.0), name='b',
                                   borrow=True)

        # create constrained parameters from uncontrained a, b
        self.theta = T.exp(self.a)
        self.phi = (self.theta / 2) * T.cos(np.pi / (self.p + 1)) * T.nnet.sigmoid(self.b)
        self.S = T.stack((self.phi, self.theta, self.phi), axis=0)

        self.c = {}  # unconstrained diagonals
        self.d = {}  # unconstrained lower triangle
        self.L = {}  # combined constrained L matrix
        self.K_mu = {}  # mean filters
        self.K_b = theano.shared(value=np.zeros((self.yDim, 1, self.yDim)).astype(theano.config.floatX),
                                 name='K_b', borrow=True,
                                 broadcastable=[False, True, False])

        for i in range(self.yDim):
            for j in range(self.yDim):
                filter_size = self.filter_size
                self.K_mu[(i, j)] = theano.shared(value=0.05 * np.random.randn(filter_size)
                                                  .astype(theano.config.floatX),
                                                  name='K_mu_%ito%i' % (i, j), borrow=True)
                self.c[(i, j)] = theano.shared(value=(-2 * np.ones(filter_size))
                                               .astype(theano.config.floatX),
                                               name='c_%ito%i' % (i, j), borrow=True)
                self.d[(i, j)] = theano.shared(value=np.zeros((filter_size, filter_size))
                                               .astype(theano.config.floatX),
                                               name='d_%ito%i' % (i, j), borrow=True)
                Ldiag = T.exp(self.c[(i, j)])  # diagonal must be positive
                self.L[(i, j)] = T.tril(self.d[(i, j)], k=-1) + T.diag(Ldiag)

    def sample_K(self):
        """
        Obtain a sample filter, K (shape = (yDim, filter_size, yDim)),
        from the gaussian distribution
        """
        K = []
        for j in range(self.yDim):  # to
            curr_filt = []
            for i in range(self.yDim):  # from
                curr_K = (self.K_mu[(i, j)] +
                          self.L[(i, j)].dot(
                          self.srng.normal((self.filter_size,))))
                curr_filt.append(curr_K)
            curr_filt = T.stack(curr_filt, axis=1)
            K.append(curr_filt)
        K = T.stack(K, axis=0)

        return K

    def evalEntropy(self):
        """
        Return the entropy term for K in the ELBO
        """
        entropy = 0
        for i in range(self.yDim):
            for j in range(self.yDim):
                Ldiag = T.diag(self.L[(i, j)])
                entropy += (self.filter_size / 2.0) * (1 + T.log(2 * np.pi))
                entropy += T.log(Ldiag).sum()
        entropy /= self.ntrials
        return entropy

    def getNextState(self, K, curr_x, curr_y):
        """
        Return predicted next data point based on given data
        """
        Upred, Ypred = self.get_UYpred(K, curr_x, curr_y, noise=True)
        Ypred = Ypred[-1]
        return Ypred

    def get_UYpred(self, K, X, Y, noise=False, pad=True):
        """
        Return the predicted U and Y for each point in Y.
        """
        if pad:
            pad = T.stack([Y[0]] * (self.filter_size - 1), axis=0)
            Y_pad = T.vertical_stack(pad, Y)

        U = T.signal.conv.conv2d(Y_pad, K[:, ::-1, ::-1],
                                 border_mode='valid').reshape((self.yDim, -1)).T
        U += X

        if noise:
            U += T.dot(self.srng.normal((self.yDim,)),
                       T.diag(self.RChol).T)

        Y_pred = Y + self.vel.reshape((1, self.yDim)) * T.tanh(U)

        return U, Y_pred

    def fit_trial(self, X, Y_true):
        '''
        Return a theano function that calculates a fit for the given data.
        '''
        K = self.sample_K()
        Upred, Ypred = self.get_UYpred(K, X, Y_true)
        return Ypred[:-1]

    def evaluateLogDensity(self, X, Y_true):
        '''
        Return a theano function that evaluates the log-density of the
        GenerativeModel.
        '''
        K = self.sample_K()
        Upred, Ypred = self.get_UYpred(K, X, Y_true)
        Upred = Upred[:-1]
        Ypred = Ypred[:-1]
        # resY = Y_true[1:] - Ypred  # get errors for each residual
        U_true = T.arctanh((Y_true[1:] - Y_true[:-1]) / self.vel.reshape((1, self.yDim)))
        resU = U_true - Upred

        # LogDensity = -(0.5*T.dot(resY.T,resY)*T.diag(self.Rinv)).sum()
        LogDensity = -(0.5*T.dot(resU.T,resU)*T.diag(self.Rinv)).sum()
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y_true.shape[0] - 0.5*(self.yDim)*np.log(2*np.pi)*Y_true.shape[0]

        curr_X = X[1:, :]
        Xpred = T.dot(X[:-1], self.A)
        resX = curr_X - Xpred
        resX0 = X[0] - self.x0
        LogDensity += (-(0.5 * T.dot(resX.T, resX) * self.Lambda).sum() -
                       0.5 * T.dot(T.dot(resX0, self.Lambda0), resX0.T))

        # noise penalties
        LogDensity += self.p * (self.Rinv).sum()
        LogDensity += self.p * T.abs_(self.Lambda).sum()
        LogDensity += self.p * T.abs_(self.Lambda0).sum()

        def sym_tridiag_det(a, b, length):
            """
            Computes the determinant of a symmetric tridiagonal
            matrix with repeating diagonals

            https://en.wikipedia.org/wiki/Tridiagonal_matrix#Determinant

            a (float): value of center diagonal
            b (float): value of up-1 and down-1 diagonals
            length (int): length of center diagonal
            """
            f_lag = 0.0
            f = 1.0
            for i in xrange(length):
                f_next = a * f - (b**2) * f_lag
                f_lag = f
                f = f_next
            return f

        filt = K - self.K_b
        conv_res = T.signal.conv.conv2d(filt, self.S.reshape((-1, 1)),
                                        border_mode='full')[:, 1:-1, :]
        LogDensity -= (0.5 * (filt * conv_res).sum()) / self.ntrials

        det_t = sym_tridiag_det(self.theta, self.phi,
                                self.filter_size)

        LogDensity += ((0.5 * T.log(T.abs_(det_t))) *
                       (self.yDim**2 / self.ntrials))

        return LogDensity + self.evalEntropy()

    def getParams(self):
        '''
        Return parameters of the GenerativeModel.
        '''
        rets = self.K_mu.values() + [self.a] + [self.b] + self.c.values()
        rets += self.d.values() + [self.K_b]
        rets += [self.RChol]
        # rets += [self.log_vel] + [self.A]
        rets += [self.QChol_diag] + [self.Q0Chol_diag]
        rets += [self.x0]
        return rets


class NNLDS(GenerativeModel):
    """
    Neural Network, Non-Linear Dynamical System
    """
    def __init__(self, GenerativeParams, xDim, yDim, yDim_in, ntrials, srng=None, nrng=None):
        super(NNLDS, self).__init__(GenerativeParams, xDim, yDim, srng, nrng)

        self.ntrials = np.cast[theano.config.floatX](ntrials)
        self.yDim_in = yDim_in  # dimension of observation input
        self.lag = GenerativeParams['lag']
        self.NN_Gen = GenerativeParams['NN_Gen']['network']
        self.PKbias_layers_gen = GenerativeParams['NN_Gen']['PKbias_layers']
        self.yCols = GenerativeParams['yCols']  # which dimensions of Y to predict
        # dynamics matrix (or NN)
        if 'NN_A' in GenerativeParams:
            self.NN_A = GenerativeParams['NN_A']['network']
            self.PKbias_layers_A = GenerativeParams['NN_A']['PKbias_layers']
            self.dyn_lag = GenerativeParams['dyn_lag']
            self.A = None
        else:
            if 'AR_P' in GenerativeParams:
                self.A = []
                for i in range(GenerativeParams['AR_P']):
                    self.A.append(theano.shared(value=np.diag(np.ones(xDim).astype(theano.config.floatX)), name='A_%i' % i, borrow=True))
            else:  # AR1
                self.A = [theano.shared(value=np.diag(np.ones(xDim).astype(theano.config.floatX)), name='A', borrow=True)]
            self.AR_P = len(self.A)  # order of AR process
        # cholesky of innovation cov matrix
        self.QChol_diag = theano.shared(value=(np.ones(xDim)).astype(theano.config.floatX), name='QChol_diag', borrow=True)
        self.QChol = T.diag(self.QChol_diag)
        # cholesky of starting distribution cov matrix
        self.Q0Chol_diag = theano.shared(value=(np.ones(xDim)).astype(theano.config.floatX), name='Q0Chol_diag', borrow=True)
        self.Q0Chol = T.diag(self.Q0Chol_diag)
        # set to zero for stationary distribution
        self.x0 = theano.shared(value=np.zeros((xDim,)).astype(theano.config.floatX), name='x0', borrow=True)
        self.trend = theano.shared(value=np.zeros((1, xDim)).astype(theano.config.floatX), name='trend',
                                   borrow=True, broadcastable=[True, False])
        # cholesky of observation noise cov matrix
        if 'RChol' in GenerativeParams:
            self.RChol = theano.shared(value=GenerativeParams['RChol'].astype(theano.config.floatX), name='RChol', borrow=True)
        else:
            self.RChol = theano.shared(value=np.random.randn(yDim).astype(theano.config.floatX) / 10, name='RChol', borrow=True)

        # we assume diagonal covariance (RChol is a vector)
        self.Rinv = 1. / (self.RChol**2)  #Tla.matrix_inverse(T.dot(self.RChol ,T.transpose(self.RChol)))
        invert = lambda mat: Tla.matrix_inverse(T.dot(mat, mat.T))
        self.Lambda = invert(self.QChol)
        self.Lambda0 = invert(self.Q0Chol)

        # learnable velocity for each observation dimension
        if 'log_vel' in GenerativeParams:
            self.log_vel = theano.shared(value=GenerativeParams['log_vel'].astype(theano.config.floatX),
                                         name='log_vel', borrow=True)
        else:
            self.log_vel = theano.shared(value=np.ones(self.yDim).astype(theano.config.floatX),
                                         name='log_vel', borrow=True)
        self.vel = T.exp(self.log_vel)

        # noise penalty constant
        if 'p_R' in GenerativeParams:
            self.p_R = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['p_R']), name='p_R', borrow=True)
        else:
            self.p_R = None
        if 'p_L' in GenerativeParams:
            self.p_L = theano.shared(value=np.cast[theano.config.floatX](GenerativeParams['p_L']), name='p_L', borrow=True)
        else:
            self.p_L = None

    def getNextState(self, curr_x, curr_y):
        """
        Return predicted next data point based on given data
        """
        Upred, Ypred = self.get_UYpred(curr_x, curr_y, noise=True,
                                       bn_update_averages=False,
                                       bn_use_averages=True)
        Ypred = Ypred[-1]
        return Ypred

    def make_lags(self, data, lag):
        """
        Take a time series and include previous time-points in each row
        """
        dims = data.shape[1]
        lag_data = data.copy()
        for i in range(1, lag + 1):
            lag_cols = T.vertical_stack(lag_data[0, :dims].reshape((1, dims)),
                                        lag_data[:-1, -dims:])
            lag_data = T.horizontal_stack(lag_data, lag_cols)

        return lag_data

    def get_UYpred(self, X, Y, noise=False, bn_use_averages=True,
                   bn_update_averages=False):
        """
        Return the predicted U and Y for each point in Y.
        """
        Y_lag = self.make_lags(Y, self.lag)

        U = lasagne.layers.get_output(self.NN_Gen, inputs=Y_lag,
                                      batch_norm_update_averages=bn_update_averages,
                                      batch_norm_use_averages=bn_use_averages)
        U += X

        if noise:
            U += T.dot(self.srng.normal((self.yDim,)),
                       T.diag(self.RChol).T)

        Y_pred = Y[:, self.yCols] + self.vel.reshape((1, self.yDim)) * T.tanh(U)

        return U, Y_pred

    def fit_trial(self, X, Y_true):
        '''
        Return a theano function that calculates a fit for the given data.
        '''
        Upred, Ypred = self.get_UYpred(X, Y_true, bn_update_averages=False,
                                       bn_use_averages=True)
        return Ypred[:-1]

    def evaluateLogDensity(self, X, Y_true):
        '''
        Return a theano function that evaluates the log-density of the
        GenerativeModel.
        '''
        Upred, Ypred = self.get_UYpred(X, Y_true, bn_update_averages=True,
                                       bn_use_averages=False)
        Upred = Upred[:-1]
        Ypred = Ypred[:-1]
        U_true = T.arctanh((Y_true[1:, self.yCols] - Y_true[:-1, self.yCols]) / self.vel.reshape((1, self.yDim)))
        resU = U_true - Upred

        LogDensity = -(0.5*T.dot(resU.T,resU)*T.diag(self.Rinv)).sum()
        LogDensity += 0.5*(T.log(self.Rinv)).sum()*Y_true.shape[0] - 0.5*(self.yDim)*np.log(2*np.pi)*Y_true.shape[0]

        if self.A is not None:
            curr_X = X[self.AR_P:, :]
            Xpred = T.zeros_like(curr_X)
            for i in range(self.AR_P - 1, -1, -1):
                Xpred += T.dot(X[i:(-self.AR_P + i)], self.A[i])
            Xpred += self.trend
        else:
            curr_X = X[1:, :]
            X_lag = self.make_lags(X[:-1], self.dyn_lag)
            Xpred = X[:-1]
            Xpred += lasagne.layers.get_output(self.NN_A, inputs=X_lag,
                                               batch_norm_update_averages=True,
                                               batch_norm_use_averages=False)
        resX = curr_X - Xpred
        resX0 = X[0] - self.x0
        LogDensity += (-(0.5 * T.dot(resX.T, resX) * self.Lambda).sum() -
                       0.5 * T.dot(T.dot(resX0, self.Lambda0), resX0.T))

        # noise penalties
        if self.p_R is not None:
            LogDensity += self.p_R * (self.Rinv).sum()
        if self.p_L is not None:
            LogDensity += self.p_L * T.abs_(self.Lambda).sum()
            LogDensity += self.p_L * T.abs_(self.Lambda0).sum()

        # prior on PK biases
        for pklayer in self.PKbias_layers_gen:
            LogDensity += pklayer.get_ELBO(self.ntrials)

        if self.A is None:
            for pklayer in self.PKbias_layers_A:
                LogDensity += pklayer.get_ELBO(self.ntrials)

        return LogDensity

    def getParams(self):
        '''
        Return parameters of the GenerativeModel.
        '''
        rets = lasagne.layers.get_all_params(self.NN_Gen, trainable=True)
        # rets += [self.log_vel]
        if self.A is None:
            rets = lasagne.layers.get_all_params(self.NN_A, trainable=True)
        else:
            rets += self.A
        rets += [self.QChol_diag] + [self.Q0Chol_diag]
        rets += [self.x0]
        # rets += [self.RChol]
        rets += [self.trend]
        return rets


class PLDS(LDS):
    '''
    Gaussian linear dynamical system with Poisson count observations. Inherits Gaussian
    linear dynamics sampling code from the LDS; implements a Poisson density evaluation
    for discrete (count) data.
    '''
    def __init__(self, GenerativeParams, xDim, yDim, srng = None, nrng = None):
        # The LDS class expects "RChol" for Gaussian observations - we just pass a dummy
        GenerativeParams['RChol'] = np.ones(1)
        super(PLDS, self).__init__(GenerativeParams,xDim,yDim,srng,nrng)

        # Currently we emulate a PLDS by having an exponential output nonlinearity.
        # Next step will be to generalize this to more flexible output nonlinearities...
        if GenerativeParams['output_nlin'] == 'exponential':
            self.rate = T.exp(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'sigmoid':
            self.rate = T.nnet.sigmoid(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        elif GenerativeParams['output_nlin'] == 'softplus':
            self.rate = T.nnet.softplus(lasagne.layers.get_output(self.NN_XtoY, inputs = self.Xsamp))
        else:
            raise Exception('Unknown output nonlinearity specification!')

    def getParams(self):
        return [self.A] + [self.QChol] + [self.Q0Chol] + [self.x0] + lasagne.layers.get_all_params(self.NN_XtoY)

    def sampleY(self):
        ''' Return a symbolic sample from the generative model. '''
        return self.srng.poisson(lam = self.rate, size = self.rate.shape)

    def sampleXY(self,_N):
        ''' Return real-valued (numpy) samples from the generative model. '''
        X = self.sampleX(_N)
        Y = np.random.poisson(lam = self.rate.eval({self.Xsamp: X}))
        return [X.astype(theano.config.floatX),Y.astype(theano.config.floatX)]

    def evaluateLogDensity(self,X,Y):
        # This is the log density of the generative model (*not* negated)
        Ypred = theano.clone(self.rate,replace={self.Xsamp: X})
        resY  = Y-Ypred
        resX  = X[1:]-T.dot(X[:-1],self.A.T)
        resX0 = X[0]-self.x0
        LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        #LatentDensity = - 0.5*T.dot(T.dot(resX0,self.Lambda0),resX0.T) - 0.5*(resX*T.dot(resX,self.Lambda)).sum() + 0.5*T.log(Tla.det(self.Lambda))*(Y.shape[0]-1) + 0.5*T.log(Tla.det(self.Lambda0)) - 0.5*(self.xDim)*np.log(2*np.pi)*Y.shape[0]
        PoisDensity = T.sum(Y * T.log(Ypred)  - Ypred - T.gammaln(Y + 1))
        LogDensity = LatentDensity + PoisDensity
        return LogDensity
