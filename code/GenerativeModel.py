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
        self.JDim = self.yDim * 2  # dimension of CGAN output
        # function that calculates states from positions
        self.get_states = GenerativeParams['get_states']

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

        # technically part of the recognition model, but it's here for
        # convenience
        self.NN_postJ_mu = GenerativeParams['NN_postJ_mu']
        self.NN_postJ_sigma = GenerativeParams['NN_postJ_sigma']

        self.yCols = GenerativeParams['yCols']  # which dimensions of Y to predict

        # velocity for each observation dimension
        self.vel = GenerativeParams['vel'].astype(theano.config.floatX)

        # coefficients for PID controller (one for each dimension)
        # https://en.wikipedia.org/wiki/PID_controller#Discrete_implementation
        unc_Kp = theano.shared(value=7*np.ones((self.yDim, 1),
                                               dtype=theano.config.floatX),
                               name='unc_Kp', borrow=True)
        unc_Ki = theano.shared(value=np.zeros((self.yDim, 1),
                                               dtype=theano.config.floatX),
                               name='unc_Ki', borrow=True)
        unc_Kd = theano.shared(value=np.zeros((self.yDim, 1),
                                               dtype=theano.config.floatX),
                               name='unc_Kd', borrow=True)

        # create list of PID controller parameters for easy access in getParams
        self.PID_params = [unc_Kp]  # [unc_Kp, unc_Ki, unc_Kd]

        # constrain PID controller parameters to be positive
        self.Kp = T.nnet.softplus(unc_Kp)
        self.Ki = T.nnet.relu(unc_Ki)  # T.nnet.softplus(unc_Ki)
        self.Kd = T.nnet.relu(unc_Kd)  # T.nnet.softplus(unc_Kd)

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
        self.unc_eps = theano.shared(value=-9 * np.ones((1, self.yDim),
                                     dtype=theano.config.floatX),
                                     name='unc_eps', borrow=True,
                                     broadcastable=[True, False])
        self.eps = T.nnet.softplus(self.unc_eps)

    def init_CGAN(self, nlayers_gen, nlayers_discr, nlayers_compress, state_dim, subID_dim, compress_dim, noise_dim,
                  hidden_dim, batch_size, compressbool,nonlinearity=leaky_rectify,
                  init_std_G=1.0, init_std_D=0.005,
                  condition_noise=None,
                  condition_scale=None, instance_noise=None,gamma=None, improveWGAN=False, lmbda=10):
        """
        Initialize Conditional Generative Adversarial Network that generates
        Gaussian mixture components, J (mu and sigma), from states and random
        noise

        This function exists so that a control model can be trained, and
        then, several cGANs can be trained using that control model.
        """
        self.CGAN_J = CGAN(nlayers_gen, nlayers_discr, state_dim, noise_dim, hidden_dim,
                           self.JDim, batch_size, self.srng, compressbool,nlayers_compress, subID_dim, compress_dim,
                           nonlinearity=nonlinearity,
                           init_std_G=init_std_G,
                           init_std_D=init_std_D,
                           condition_noise=condition_noise,
                           condition_scale=condition_scale,
                           instance_noise=instance_noise,gamma=gamma, improveWGAN=improveWGAN, lmbda=lmbda)

    def init_GAN(self, nlayers_gen, nlayers_discr, noise_dim,
                 hidden_dim, batch_size, nonlinearity=leaky_rectify,
                 init_std_G=1.0, init_std_D=0.005,
                 instance_noise=None, cond_dim=0):
        """
        Initialize Generative Adversarial Network that generates
        initial goal state, g_0, from random noise (and, optionally,
        conditions such as subject ID)

        This function exists so that a control model can be trained, and
        then, several GANs can be trained using that control model.
        """
        if cond_dim > 0:
            self.GAN_g0 = CGAN(nlayers_gen, nlayers_discr, cond_dim, noise_dim,
                               hidden_dim, self.yDim, batch_size, self.srng, 
                               nonlinearity=nonlinearity,
                               init_std_G=init_std_G,
                               init_std_D=init_std_D,
                               instance_noise=instance_noise)
            self.g0_extra_conds = True
        else:
            self.GAN_g0 = WGAN(nlayers_gen, nlayers_discr, noise_dim,
                               hidden_dim, self.yDim, batch_size, self.srng,
                               nonlinearity=nonlinearity,
                               init_std_G=init_std_G,
                               init_std_D=init_std_D,
                               instance_noise=instance_noise)
            self.g0_extra_conds = False

    def get_preds(self, Y, U, subIDconds, training=False, post_g=None, postJ=None,
                  gen_g=None, extra_conds=None):
        """
        Return the predicted next J, g, and U for each point in Y. (Plus U at t=0)

        For training: provide postJ and post_g, samples from the posterior,
                      which are used to calculate the ELBO
        For generating new data: provide gen_g, the generated goal states up to
                                 the current timepoint
        """
        if training and (post_g is None or postJ is None):
            raise Exception(
                "Must provide samples from posteriors during training")
        # Draw next goals based on force
        if postJ is not None and post_g is not None:
            J = None  # not generating J from CGAN, using sample from posterior
            J_mean = postJ[:, :self.yDim]
            J_scale = T.nnet.softplus(postJ[:, self.yDim:])
            next_g = (post_g[:-1] + J_scale * J_mean) / (1 + J_scale)
        elif gen_g is not None:
            # get states from position
            states = self.get_states(Y)
            if extra_conds is not None:
                states = T.horizontal_stack(states, extra_conds.astype(theano.config.floatX))
            # Get external force from CGAN
            J = self.CGAN_J.get_generated_data(states, subIDconds, training=training)
            J_mean = J[:, :self.yDim]
            J_scale = T.nnet.softplus(J[:, self.yDim:])
            goal = ((gen_g[(-1,)] + J_scale[(-1,)] * J_mean[(-1,)]) /
                    (1 + J_scale[(-1,)]))
            var = self.sigma**2 / (1 + J_scale[(-1,)])
            goal += self.srng.normal(goal.shape) * T.sqrt(var)
            next_g = T.vertical_stack(gen_g[1:],
                                      goal)
        else:
            raise Exception("Goal states must be provided " +
                            "(either posterior or generated)")
        # PID Controller for next control point
        if post_g is not None:
            error = post_g - T.vertical_stack(Y[[[0]], self.yCols], Y[:, self.yCols])
        else:
            error = next_g - Y[:, self.yCols]

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
        if post_g is None:
            Udiff += self.eps * self.srng.normal(Udiff.shape)
            Upred = U + Udiff
        else:
            Upred = T.vertical_stack(T.zeros((1, self.yDim)), U) + Udiff

        return J, next_g, Upred

    def getNextState(self, curr_y, curr_u, curr_g, extra_conds=None):
        """
        Generate predicted next data point based on given data.
        Used for generating trials. We keep track of g externally because it
        is dependent on the previous g.
        """
        if self.CGAN_J is None:
            raise Exception("Must initiate and train CGAN before calling")
        _, g_pred, Upred = self.get_preds(curr_y, curr_u, gen_g=curr_g,
                                             extra_conds=extra_conds)
        return g_pred[-1], Upred[-1]

    def fit_trial(self, g, Y_true, U_true):
        '''
        Return a theano expression that calculates a fit (next timestep
        prediction) for the given data.
        '''
        self.draw_postJ(g)
        _, _, Upred = self.get_preds(Y_true[:-1],
                                     U_true[:-1],
                                     training=False,
                                     post_g=g,
                                     postJ=self.postJ)
        return Upred

    def draw_postJ(self, g):
        """
        Calculate posterior of J using current and next goal
        """
        # get current and next goal
        g_stack = T.horizontal_stack(g[:-1], g[1:])
        self.postJ_mu = lasagne.layers.get_output(self.NN_postJ_mu,
                                                  inputs=g_stack)
        batch_unc_sigma = lasagne.layers.get_output(self.NN_postJ_sigma,
                                                    inputs=g_stack).reshape(
                                                        (-1, self.JDim,
                                                         self.JDim))

        def constrain_sigma(unc_sigma):
            return (T.diag(T.nnet.softplus(T.diag(unc_sigma))) +
                    T.tril(unc_sigma, k=-1))

        self.postJ_sigma, _ = theano.scan(fn=constrain_sigma,
                                          outputs_info=None,
                                          sequences=[batch_unc_sigma])
        self.postJ = self.postJ_mu + T.batched_dot(self.postJ_sigma,
                                                   self.srng.normal(
                                                       (g_stack.shape[0],
                                                        self.JDim)))

    def evaluateGANLoss(self, post_g0, g0_conds=None, mode='D'):
        """
        Evaluate loss of GAN
        Mode is D for discriminator, G for generator
        """
        if self.GAN_g0 is None:
            raise Exception("Must initiate GAN before calling")
        if self.g0_extra_conds:
            # Get external force from CGAN
            gen_g0 = self.GAN_g0.get_generated_data(g0_conds,
                                                    training=True)
            if mode == 'D':
                return self.GAN_g0.get_discr_cost(post_g0, gen_g0, g0_conds)
            elif mode == 'G':
                return self.GAN_g0.get_gen_cost(gen_g0, g0_conds)
            else:
                raise Exception("Invalid mode. Provide 'G' for generator loss " +
                                "or 'D' for discriminator loss.")
        else:
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

    def evaluateCGANLoss(self, postJ, states, subID, mode='D'):
        """
        Evaluate loss of cGAN
        Mode is D for discriminator, G for generator
        """
        if self.CGAN_J is None:
            raise Exception("Must initiate cGAN before calling")
        # Get external force from CGAN
        #import pdb; pdb.set_trace()
        # if self.CGAN_J.compressbool is False:
        # 	genJ = self.CGAN_J.get_generated_data(states, compress=False,subIDconds=subID, training=True)
        # else:
        genJ = self.CGAN_J.get_generated_data(states, subIDconds=subID, training=True)

        if mode == 'D':
            return self.CGAN_J.get_discr_cost(postJ, genJ,
                                              states, subID)
        elif mode == 'G':
            return self.CGAN_J.get_gen_cost(genJ, states, subID)
        else:
            raise Exception("Invalid mode. Provide 'G' for generator loss " +
                            "or 'D' for discriminator loss.")

    def evaluateLogDensity(self, g, Y, U, subID):
        '''
        Return a theano function that evaluates the log-density of the
        GenerativeModel.

        g: Goal state time series (sample from the recognition model)
        Y: Time series of positions
        '''
        # get q(J|g)
        self.draw_postJ(g)
        # Calculate real control signal
        #U_true = T.arctanh((Y[1:, self.yCols] - Y[:-1, self.yCols]) /
        #                   self.vel.reshape((1, self.yDim)))

        # Get predictions for next timestep (at each timestep except for last)
        # disregard last timestep bc we don't know the next value, thus, we
        # can't calculate the error
        Jpred, g_pred, Upred = self.get_preds(Y[:-1],
                                              U[:-1], subID,
                                              training=True,
                                              post_g=g,
                                              postJ=self.postJ)
        # calculate loss on control signal
        resU = U - Upred
        LogDensity = -(resU**2 / (2 * self.eps**2)).sum()
        LogDensity -= 0.5 * T.log(2 * np.pi) + T.log(self.eps).sum()

        # calculate loss on goal state
        res_g = g[1:] - g_pred
        LogDensity -= (res_g**2 / (2 * self.sigma**2)).sum()
        LogDensity -= 0.5 * T.log(2 * np.pi) + T.log(self.sigma).sum()

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
        if self.pen_sigma is not None:
            LogDensity -= self.pen_sigma * self.unc_sigma.sum()

        return LogDensity

    def getParams(self):
        '''
        Return the learnable parameters of the model
        '''
        rets = lasagne.layers.get_all_params(self.NN_postJ_mu)
        rets += lasagne.layers.get_all_params(self.NN_postJ_sigma)
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
