"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer and 2017 Shariq Iqbal

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
from theano.tensor.signal import conv
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from CGAN import CGAN, WGAN
from lasagne.nonlinearities import leaky_rectify

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

def logsumexp(x, axis=None):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max

class GBDS(GenerativeModel):
    """
    Goal-Based Dynamical System

    Inputs:
    - GenerativeParams: A dictionary of parameters for the model
        Entries include:
        - get_states: function that calculates current state from position
        - pen_eps: Penalty on control signal noise, epsilon
        - pen_sigma: Penalty on goals state noise, sigma
        - pen_g: Two penalties on goal state leaving boundaries (Can be set
                 None)
        - bounds_g: Boundaries corresponding to above penalties
        - NN_postJ_mu: Neural network that parametrizes the mean of the
                       posterior of J (i.e. mu and sigma), conditioned on goals
        - NN_postJ_sigma: Neural network that parametrizes the covariance of the
                          posterior of J (i.e. mu and sigma), conditioned on goals 
        - yCols: Columns of Y this agent corresponds to. Used to index columns
                 in real data to compare against generated data.
        - vel: Maximum velocity of each dimension in yCols.
    - yDim: Number of dimensions for this agent
    - yDim_in: Numver of total dimensions in the data
    - srng: Theano symbolic random number generator (theano RandomStreams
            object)
    - nrng: Numpy random number generator
    """
    def __init__(self, GenerativeParams, yDim, yDim_in,
                 srng=None, nrng=None):
        super(GBDS, self).__init__(GenerativeParams, yDim, yDim, srng, nrng)
        self.yDim_in = yDim_in  # dimension of observation input
        # function that calculates states from positions
        self.get_states = GenerativeParams['get_states']
        
        # GMM networks
        self.GMM_k = GenerativeParams['GMM_k']  # number of GMM components
        self.GMM_net = GenerativeParams['GMM_net']
        self.GMM_mu = lasagne.layers.SliceLayer(self.GMM_net, axis=1, indices=slice(None, yDim * self.GMM_k))
        self.GMM_lambda = lasagne.layers.NonlinearityLayer(
            lasagne.layers.SliceLayer(
                self.GMM_net, axis=1,
                indices=slice(yDim * self.GMM_k, 2 * yDim * self.GMM_k)),
            nonlinearity=lasagne.nonlinearities.softplus)
        self.GMM_w = lasagne.layers.NonlinearityLayer(
            lasagne.layers.SliceLayer(
                self.GMM_net, axis=1,
                indices=slice(2 * yDim * self.GMM_k, None)),
            nonlinearity=lasagne.nonlinearities.softmax)

        # penalty on epsilon (noise on control signal)
        if 'pen_eps' in GenerativeParams:
            self.pen_eps = GenerativeParams['pen_eps']
        else:
            self.pen_eps = None

        # penalty on sigma (noise on goal state)
        if 'pen_sigma' in GenerativeParams:
            self.pen_sigma = GenerativeParams['pen_sigma']
        else:
            self.pen_sigma = None

        # penalties on goal state passing boundaries
        if 'pen_g' in GenerativeParams:
            self.pen_g = GenerativeParams['pen_g']
        else:
            self.pen_g = (None, None)

        # corresponding boundaries for pen_g
        if 'bounds_g' in GenerativeParams:
            self.bounds_g = GenerativeParams['bounds_g']
        else:
            self.bounds_g = (1.0, 1.5)

        self.yCols = GenerativeParams['yCols']  # which dimensions of Y to predict
        
        # velocity for each observation dimension (of all agents)
        self.all_vel = GenerativeParams['all_vel'].astype(theano.config.floatX)
        # velocity for each observation dimension (of this agent)
        self.vel = GenerativeParams['vel'].astype(theano.config.floatX)

        # coefficients for PID controller (one for each dimension)
        # https://en.wikipedia.org/wiki/PID_controller#Discrete_implementation
        unc_Kp = theano.shared(value=np.zeros((self.yDim, 1),
                                              dtype=theano.config.floatX),
                               name='unc_Kp', borrow=True)
        unc_Ki = theano.shared(value=np.zeros((self.yDim, 1),
                                              dtype=theano.config.floatX),
                               name='unc_Ki', borrow=True)
        unc_Kd = theano.shared(value=np.zeros((self.yDim, 1),
                                              dtype=theano.config.floatX),
                               name='unc_Kd', borrow=True)

        # create list of PID controller parameters for easy access in getParams
        # self.PID_params = [unc_Kp]
        self.PID_params = [unc_Kp, unc_Ki, unc_Kd]

        # constrain PID controller parameters to be positive
        self.Kp = T.nnet.softplus(unc_Kp)
        # self.Ki = unc_Ki
        self.Ki = T.nnet.softplus(unc_Ki)
        # self.Kd = unc_Kd
        self.Kd = T.nnet.softplus(unc_Kd)

        # calculate coefficients to be placed in convolutional filter
        t_coeff = self.Kp + self.Ki + self.Kd
        t1_coeff = -self.Kp - 2 * self.Kd
        t2_coeff = self.Kd

        # concatenate coefficients into filter
        self.L = T.horizontal_stack(t_coeff, t1_coeff, t2_coeff)

        # noise coefficient on goal states
        self.unc_sigma = theano.shared(value=-7 * np.ones((1, self.yDim),
                                       dtype=theano.config.floatX),
                                       name='unc_sigma', borrow=True,
                                       broadcastable=[True, False])
        self.sigma = T.nnet.softplus(self.unc_sigma)

        # noise coefficient on control signals
        self.unc_eps = theano.shared(value=-11 * np.ones((1, self.yDim),
                                     dtype=theano.config.floatX),
                                     name='unc_eps', borrow=True,
                                     broadcastable=[True, False])
        self.eps = T.nnet.softplus(self.unc_eps)

    def sample_GMM(self, states):
        """
        Sample from GMM based on highest weight component
        """
        mu = lasagne.layers.get_output(self.GMM_mu, inputs=states).reshape(
            (-1, self.GMM_k, self.yDim))
        lmbda = lasagne.layers.get_output(self.GMM_lambda, inputs=states).reshape(
            (-1, self.GMM_k, self.yDim))
        all_w = lasagne.layers.get_output(self.GMM_w, inputs=states)
        def select_components(sub_mu, sub_lmbda, w):
            component = self.srng.choice(size=[1], a=T.arange(self.GMM_k), p=w.flatten())  # weighted sample of index
            return sub_mu[component[0], :], sub_lmbda[component[0], :]
        (mu_k, lmbda_k), updates = theano.scan(fn=select_components,
                                               outputs_info=None,
                                               sequences=[mu, lmbda, all_w])
        return (mu, lmbda, all_w, mu_k, lmbda_k), updates

    def get_preds(self, Y, training=False, post_g=None,
                  gen_g=None, extra_conds=None):
        """
        Return the predicted next J, g, U, and Y for each point in Y.

        For training: provide post_g, sample from the posterior,
                      which is used to calculate the ELBO
        For generating new data: provide gen_g, the generated goal states up to
                                 the current timepoint
        """
        if training and post_g is None:
            raise Exception(
                "Must provide sample of g from posterior during training")
        # get states from position
        states = self.get_states(Y, max_vel=self.all_vel)
        if extra_conds is not None:
            states = T.horizontal_stack(states, extra_conds)
        (all_mu, all_lmbda, all_w, mu_k, lmbda_k), updates = self.sample_GMM(states)
        # Draw next goals based on force
        if post_g is not None:  # Calculate next goals from posterior
            next_g = ((post_g[:-1].reshape((-1, 1, self.yDim)) + all_mu * all_lmbda) / (1 + all_lmbda))
        elif gen_g is not None:  # Generate next goals
            # Get external force from GMM
            goal = ((gen_g[(-1,)] + lmbda_k[(-1,)] * mu_k[(-1,)]) /
                    (1 + lmbda_k[(-1,)]))
            var = self.sigma**2 / (1 + lmbda_k[(-1,)])
            goal += self.srng.normal(goal.shape) * T.sqrt(var)
            next_g = T.vertical_stack(gen_g[1:],
                                      goal)
        else:
            raise Exception("Goal states must be provided " +
                            "(either posterior or generated)")
        # PID Controller for next control point
        if post_g is not None:  # calculate error from posterior goals
            error = post_g[1:] - Y[:, self.yCols]
        else:  # calculate error from generated goals
            error = next_g - Y[:, self.yCols]

        # Assume control starts at zero
        Uprev = T.vertical_stack(T.zeros((1, self.yDim)),
                                 T.arctanh((Y[1:, self.yCols] -
                                           Y[:-1, self.yCols]) /
                                 self.vel.reshape((1, self.yDim))))
        Udiff = []
        for i in range(self.yDim):
            # get current error signal and corresponding filter
            signal = error[:, i]
            filt = self.L[i].reshape((1, 1, -1, 1))
            # zero pad beginning
            signal = T.concatenate((T.zeros(2), signal)).reshape((1, 1, -1, 1))
            res = T.nnet.conv2d(signal, filt, border_mode='valid')
            res = res.reshape((-1, 1))
            Udiff.append(res)
        if len(Udiff) > 1:
            Udiff = T.horizontal_stack(*Udiff)
        else:
            Udiff = Udiff[0]
        if post_g is None:  # Add control signal noise to generated data
            Udiff += self.eps * self.srng.normal(Udiff.shape)
        Upred = Uprev + Udiff
        # get predicted Y
        Ypred = Y[:, self.yCols] + self.vel.reshape((1, self.yDim)) * T.tanh(Upred)

        return (all_mu, all_lmbda, all_w, next_g, Upred, Ypred), updates

    def evaluateLogDensity(self, g, Y):
        '''
        Return a theano function that evaluates the log-density of the
        GenerativeModel.

        g: Goal state time series (sample from the recognition model)
        Y: Time series of positions
        '''
        # Calculate real control signal
        U_true = T.arctanh((Y[1:, self.yCols] - Y[:-1, self.yCols]) /
                           self.vel.reshape((1, self.yDim)))
        # Get predictions for next timestep (at each timestep except for last)
        # disregard last timestep bc we don't know the next value, thus, we
        # can't calculate the error
        (all_mu, all_lmbda, all_w, g_pred, Upred, Ypred), updates = self.get_preds(
            Y[:-1],
            training=True,
            post_g=g)
        # calculate loss on control signal
        resU = U_true - Upred
        LogDensity = -(resU**2 / (2 * self.eps**2)).sum()
        LogDensity -= 0.5 * T.log(2 * np.pi) + T.log(self.eps).sum()
        
        # calculate GMM loss
        w_brdcst = all_w.reshape((-1, self.GMM_k, 1))
        gmm_res_g = g[1:].reshape((-1, 1, self.yDim)) - g_pred
        gmm_term = T.log(w_brdcst + 1e-8) - ((1 + all_lmbda) / (2 * self.sigma.reshape((1,1,-1))**2)) * gmm_res_g**2
        gmm_term += 0.5 * T.log(1 + all_lmbda) - 0.5 * T.log(2 * np.pi) - T.log(self.sigma.reshape((1,1,-1)))
        # LogDensity += T.log(T.exp(gmm_term).sum(axis=1)).sum()
        LogDensity += logsumexp(gmm_term.sum(axis=2), axis=1).sum()

        # linear penalty on goal state escaping game space
        if self.pen_g[0] is not None:
            LogDensity -= self.pen_g[0] * T.nnet.relu(g_pred - self.bounds_g[0]).sum()
            LogDensity -= self.pen_g[0] * T.nnet.relu(-g_pred - self.bounds_g[0]).sum()
        if self.pen_g[1] is not None:
            LogDensity -= self.pen_g[1] * T.nnet.relu(g_pred - self.bounds_g[1]).sum()
            LogDensity -= self.pen_g[1] * T.nnet.relu(-g_pred - self.bounds_g[1]).sum()

        # penalty on eps
        if self.pen_eps is not None:
            LogDensity -= self.pen_eps * self.unc_eps.sum()

        # penalty on sigma
        # if self.pen_sigma is not None:
        #     LogDensity -= self.pen_sigma * self.unc_sigma.sum()

        return LogDensity, updates
    
    def init_GAN(self, *args, **kwargs):
        """
        Initialize Generative Adversarial Network that generates
        initial goal state, g_0, from random noise
        Look at CGAN.py for initialization parameters.
        This function exists so that a control model can be trained, and
        then, several GANs can be trained using that control model.
        """
        kwargs['srng'] = self.srng
        self.GAN_g0 = WGAN(*args, **kwargs)
    
    def evaluateGANLoss(self, post_g0, mode='D'):
        """
        Evaluate loss of GAN
        Mode is D for discriminator, G for generator
        """
        if self.GAN_g0 is None:
            raise Exception("Must initiate GAN before calling")
        # Get external force from CGAN
        gen_g0 = self.GAN_g0.get_generated_data(post_g0.shape[0],
                                                training=True)
        if mode == 'D':
            return self.GAN_g0.get_discr_cost(post_g0, gen_g0)
        elif mode == 'G':
            return self.GAN_g0.get_gen_cost(gen_g0)
        else:
            raise Exception("Invalid mode. Provide 'G' for generator loss " +
                            "or 'D' for discriminator loss.")

    def getParams(self):
        '''
        Return the learnable parameters of the model
        '''
        rets = lasagne.layers.get_all_params(self.GMM_net)
        rets += self.PID_params + [self.unc_eps]  #+ [self.unc_sigma]
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
