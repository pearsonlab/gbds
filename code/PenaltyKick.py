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
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import numpy as np
from numpy.random import *
import sys

sys.path.append('./lib/')
#from Trainable import *
from GenerativeModel import *
from RecognitionModel import *

class SGVB_PK():#(Trainable):
    '''
    This class is the same as the SGVB class, but with two separate generative models
    for the ball and the goalie.

    Inputs:
    gen_params_ball       - Dictionary of parameters that define the chosen GenerativeModel
    GEN_MODEL_ball        - A class that inhereits from the GenerativeModel abstract class
    gen_params_goalie     - Dictionary of parameters that define the chosen GenerativeModel
    GEN_MODEL_goalie      - A class that inhereits from the GenerativeModel abstract class
    rec_params            - Dictionary of parameters that define the chosen RecognitionModel
    REC_MODEL             - A class that inhereits from the RecognitionModel abstract class
    xDim                  - Integer that specifies the dimensionality of the latent space
    yDim                  - Integer that specifies the dimensionality of the observations

    --------------------------------------------------------------------------
    This code is a reference implementation of the algorithm described in:
    * Black box variational inference for state space models
           - Archer et al (arxiv preprint, 2015)  [http://arxiv.org/abs/1511.07367]

    The SGVB ("Stochastic Gradient Variational Bayes") inference technique is described
    in the following publications:
    * Auto-Encoding Variational Bayes
           - Kingma, Welling (ICLR, 2014)
    * Stochastic backpropagation and approximate inference in deep generative models.
           - Rezende et al (ICML, 2014)
    * Doubly stochastic variational bayes for non-conjugate inference.
           - Titsias and Lazaro-Gredilla (ICML, 2014)
    '''
    def __init__(self,
                 gen_params_ball,  # dictionary of generative model parameters
                 GEN_MODEL_ball,  # class that inherits from GenerativeModel
                 gen_params_goalie,  # dictionary of generative model parameters
                 GEN_MODEL_goalie,  # class that inherits from GenerativeModel
                 rec_params,  # dictionary of approximate posterior ("recognition model") parameters
                 REC_MODEL,  # class that inherits from RecognitionModel
                 yCols_goalie=(0, 3, 6),  # columns holding goalie observations
                 xCols_goalie=(0,),  # columns holding goalie latents
                 yCols_ball=(1, 2, 4, 5, 7, 8),  # columns holding ball observations
                 xCols_ball=(1,),  # columns holding ball latents
                 feed_external=False  # feed external observations to each gen model
                 ):

        # instantiate rng's
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)

        #---------------------------------------------------------
        ## actual model parameters
        self.X, self.Y = T.matrices('X', 'Y')   # symbolic variables for the data

        self.xDim_goalie = len(xCols_goalie)
        self.yDim_goalie = len(yCols_goalie)

        self.xDim_ball = len(xCols_ball)
        self.yDim_ball = len(yCols_ball)

        self.xDim = self.xDim_goalie + self.xDim_ball
        self.yDim = self.yDim_goalie + self.yDim_ball

        self.yCols_goalie = yCols_goalie
        self.xCols_goalie = xCols_goalie
        self.yCols_ball = yCols_ball
        self.xCols_ball = xCols_ball

        self.feed_external = feed_external

        # instantiate our prior & recognition models
        self.mrec = REC_MODEL(rec_params, self.Y, self.xDim,
                              self.yDim, self.srng, self.nrng)
        if self.feed_external:
            self.mprior_ball = GEN_MODEL_ball(gen_params_ball, self.xDim_ball,
                                              self.yDim_ball, srng=self.srng,
                                              nrng=self.nrng, y_extDim=self.yDim_goalie)
            self.mprior_goalie = GEN_MODEL_goalie(gen_params_goalie, self.xDim_goalie,
                                                  self.yDim_goalie, srng=self.srng,
                                                  nrng=self.nrng, y_extDim=self.yDim_ball)
        else:
            self.mprior_ball = GEN_MODEL_ball(gen_params_ball, self.xDim_ball,
                                              self.yDim_ball, srng=self.srng,
                                              nrng=self.nrng)
            self.mprior_goalie = GEN_MODEL_goalie(gen_params_goalie, self.xDim_goalie,
                                                  self.yDim_goalie, srng=self.srng,
                                                  nrng=self.nrng)

        self.isTrainingRecognitionModel = True
        self.isTrainingGenerativeModel = True

    def getParams(self):
        '''
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []
        if self.isTrainingRecognitionModel:
            params = params + self.mrec.getParams()
        if self.isTrainingGenerativeModel:
            params = params + self.mprior_ball.getParams()
            params = params + self.mprior_goalie.getParams()
        return params

    def EnableRecognitionModelTraining(self):
        '''
        Enable training of RecognitionModel parameters.
        '''
        self.isTrainingRecognitionModel = True;
        self.mrec.setTrainingMode()

    def DisableRecognitionModelTraining(self):
        '''
        Disable training of RecognitionModel parameters.
        '''
        self.isTrainingRecognitionModel = False;
        self.mrec.setTestMode()

    def EnableGenerativeModelTraining(self):
        '''
        Enable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = True;
        print('Enable switching training/test mode in generative model class!\n')
    def DisableGenerativeModelTraining(self):
        '''
        Disable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = False;
        print('Enable switching training/test mode in generative model class!\n')

    def cost(self):
        '''
        Compute a one-sample approximation the ELBO (lower bound on marginal likelihood), normalized by batch size (length of Y in first dimension).
        '''
        q = self.mrec.getSample()

        theentropy = self.mrec.evalEntropy()

        if self.feed_external:
            thelik = self.mprior_goalie.evaluateLogDensity(q[:, self.xCols_goalie].reshape((-1, self.xDim_goalie)),
                                                           self.Y[:, self.yCols_goalie],
                                                           Y_ext=self.Y[:, self.yCols_ball])
            thelik += self.mprior_ball.evaluateLogDensity(q[:, self.xCols_ball].reshape((-1, self.xDim_ball)),
                                                          self.Y[:, self.yCols_ball],
                                                          Y_ext=self.Y[:, self.yCols_goalie])
        else:
            thelik = self.mprior_goalie.evaluateLogDensity(q[:, self.xCols_goalie].reshape((-1, self.xDim_goalie)),
                                                           self.Y[:, self.yCols_goalie])
            thelik += self.mprior_ball.evaluateLogDensity(q[:, self.xCols_ball].reshape((-1, self.xDim_ball)),
                                                          self.Y[:, self.yCols_ball])

        thecost = thelik + theentropy

        return thecost/self.Y.shape[0]


class SGVB_PK_simp():#(Trainable):
    '''
    This class is the same as the SGVB_PK class, but with no latents

    Inputs:
    gen_params_ball       - Dictionary of parameters that define the chosen GenerativeModel
    GEN_MODEL_ball        - A class that inhereits from the GenerativeModel abstract class
    gen_params_goalie     - Dictionary of parameters that define the chosen GenerativeModel
    GEN_MODEL_goalie      - A class that inhereits from the GenerativeModel abstract class
    yDim                  - Integer that specifies the dimensionality of the observations

    --------------------------------------------------------------------------
    This code is a reference implementation of the algorithm described in:
    * Black box variational inference for state space models
           - Archer et al (arxiv preprint, 2015)  [http://arxiv.org/abs/1511.07367]

    The SGVB ("Stochastic Gradient Variational Bayes") inference technique is described
    in the following publications:
    * Auto-Encoding Variational Bayes
           - Kingma, Welling (ICLR, 2014)
    * Stochastic backpropagation and approximate inference in deep generative models.
           - Rezende et al (ICML, 2014)
    * Doubly stochastic variational bayes for non-conjugate inference.
           - Titsias and Lazaro-Gredilla (ICML, 2014)
    '''
    def __init__(self,
                 gen_params_ball,  # dictionary of generative model parameters
                 GEN_MODEL_ball,  # class that inherits from GenerativeModel
                 gen_params_goalie,  # dictionary of generative model parameters
                 GEN_MODEL_goalie,  # class that inherits from GenerativeModel
                 yCols_goalie=(0, 3, 6),  # columns holding goalie observations
                 yCols_ball=(1, 2, 4, 5, 7, 8),  # columns holding ball observations
                 feed_external=False  # feed external observations to each gen model
                 ):

        # instantiate rng's
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)

        #---------------------------------------------------------
        ## actual model parameters
        self.Y = T.matrix('Y')   # symbolic variables for the data

        self.yDim_goalie = len(yCols_goalie)
        self.yDim_ball = len(yCols_ball)

        self.yDim = self.yDim_goalie + self.yDim_ball

        self.yCols_goalie = yCols_goalie
        self.yCols_ball = yCols_ball

        self.feed_external = feed_external

        # instantiate our prior model
        if self.feed_external:
            self.mprior_ball = GEN_MODEL_ball(gen_params_ball,
                                              self.yDim_ball, srng=self.srng,
                                              nrng=self.nrng, y_extDim=self.yDim_goalie)
            self.mprior_goalie = GEN_MODEL_goalie(gen_params_goalie,
                                                  self.yDim_goalie, srng=self.srng,
                                                  nrng=self.nrng, y_extDim=self.yDim_ball)
        else:
            self.mprior_ball = GEN_MODEL_ball(gen_params_ball,
                                              self.yDim_ball, srng=self.srng,
                                              nrng=self.nrng)
            self.mprior_goalie = GEN_MODEL_goalie(gen_params_goalie,
                                                  self.yDim_goalie, srng=self.srng,
                                                  nrng=self.nrng)

        self.isTrainingGenerativeModel = True

    def getParams(self):
        '''
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []
        if self.isTrainingGenerativeModel:
            params = params + self.mprior_ball.getParams()
            params = params + self.mprior_goalie.getParams()
        return params

    def EnableGenerativeModelTraining(self):
        '''
        Enable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = True;
        print('Enable switching training/test mode in generative model class!\n')
    def DisableGenerativeModelTraining(self):
        '''
        Disable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = False;
        print('Enable switching training/test mode in generative model class!\n')

    def cost(self):
        '''
        Compute a one-sample approximation the ELBO (lower bound on marginal likelihood), normalized by batch size (length of Y in first dimension).
        '''

        if self.feed_external:
            thelik = self.mprior_goalie.evaluateLogDensity(self.Y[:, self.yCols_goalie],
                                                           Y_ext=self.Y[:, self.yCols_ball])
            thelik += self.mprior_ball.evaluateLogDensity(self.Y[:, self.yCols_ball],
                                                          Y_ext=self.Y[:, self.yCols_goalie])
        else:
            thelik = self.mprior_goalie.evaluateLogDensity(self.Y[:, self.yCols_goalie])
            thelik += self.mprior_ball.evaluateLogDensity(self.Y[:, self.yCols_ball])

        return thelik/self.Y.shape[0]

class SGVB_PK_GP():#(Trainable):
    '''
    This class fits a gaussian process model to PenaltyKick data.

    Inputs:
    gen_params       - Dictionary of parameters that define the chosen GenerativeModel
    GEN_MODEL        - A class that inhereits from the GenerativeModel abstract class
    yDim             - Integer that specifies the dimensionality of the observations

    --------------------------------------------------------------------------

    The SGVB ("Stochastic Gradient Variational Bayes") inference technique is described
    in the following publications:
    * Auto-Encoding Variational Bayes
           - Kingma, Welling (ICLR, 2014)
    * Stochastic backpropagation and approximate inference in deep generative models.
           - Rezende et al (ICML, 2014)
    * Doubly stochastic variational bayes for non-conjugate inference.
           - Titsias and Lazaro-Gredilla (ICML, 2014)
    '''
    def __init__(self,
                 gen_params,  # dictionary of generative model parameters
                 GEN_MODEL,  # class that inherits from GenerativeModel
                 yDim,  # number of observation dimensions
                 ntrials  # total number of trials in training set
                 ):

        # instantiate rng's
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)

        #---------------------------------------------------------
        ## actual model parameters
        self.Y = T.matrix('Y')   # symbolic variables for the data

        self.yDim = yDim

        # instantiate our prior model
        self.mprior = GEN_MODEL(gen_params,
                                self.yDim, ntrials, srng=self.srng,
                                nrng=self.nrng)

        self.isTrainingGenerativeModel = True

    def getParams(self):
        '''
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []
        if self.isTrainingGenerativeModel:
            params = params + self.mprior.getParams()
        return params

    def EnableGenerativeModelTraining(self):
        '''
        Enable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = True;
        print('Enable switching training/test mode in generative model class!\n')

    def DisableGenerativeModelTraining(self):
        '''
        Disable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = False;
        print('Enable switching training/test mode in generative model class!\n')

    def cost(self):
        '''
        Compute a one-sample approximation the ELBO (lower bound on marginal likelihood), normalized by batch size (length of Y in first dimension).
        '''

        thelik = self.mprior.evaluateLogDensity(self.Y)
        return thelik / self.Y.shape[0]


class SGVB_PK_GP2():#(Trainable):
    '''
    This class fits a gaussian process model to PenaltyKick data.

    Inputs:
    gen_params       - Dictionary of parameters that define the chosen GenerativeModel
    GEN_MODEL        - A class that inhereits from the GenerativeModel abstract class
    yDim             - Integer that specifies the dimensionality of the observations

    --------------------------------------------------------------------------

    The SGVB ("Stochastic Gradient Variational Bayes") inference technique is described
    in the following publications:
    * Auto-Encoding Variational Bayes
           - Kingma, Welling (ICLR, 2014)
    * Stochastic backpropagation and approximate inference in deep generative models.
           - Rezende et al (ICML, 2014)
    * Doubly stochastic variational bayes for non-conjugate inference.
           - Titsias and Lazaro-Gredilla (ICML, 2014)
    '''
    def __init__(self,
                 gen_params,  # dictionary of generative model parameters
                 GEN_MODEL,  # class that inherits from GenerativeModel
                 yDim,  # number of observation dimensions
                 xDim_goalie,
                 xDim_ball,
                 rec_params_goalie, # dictionary of approximate posterior ("recognition model") parameters
                 rec_params_ball, # dictionary of approximate posterior ("recognition model") parameters
                 REC_MODEL,
                 ntrials,  # total number of trials in training set
                 noX=False
                 ):

        # instantiate rng's
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)

        #---------------------------------------------------------
        ## actual model parameters
        self.noX = noX
        self.X, self.Y = T.matrices('X', 'Y')   # symbolic variables for the data

        self.xDim_goalie = xDim_goalie
        self.xDim_ball = xDim_ball
        self.yDim = yDim

        # instantiate our prior and recognition models
        if not self.noX:
            self.mrec_goalie = REC_MODEL(rec_params_goalie, self.Y,
                                         self.xDim_goalie, self.yDim,
                                         self.srng, self.nrng)
            self.mrec_ball = REC_MODEL(rec_params_ball, self.Y,
                                       self.xDim_ball, self.yDim,
                                       self.srng, self.nrng)
            self.isTrainingRecognitionModel = True
        else:
            self.isTrainingRecognitionModel = False
        self.mprior = GEN_MODEL(gen_params, (self.xDim_goalie, self.xDim_ball),
                                self.yDim, ntrials, srng=self.srng,
                                nrng=self.nrng, noX=noX)

        self.isTrainingGenerativeModel = True

    def getParams(self):
        '''
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []
        if self.isTrainingRecognitionModel:
            params += self.mrec_goalie.getParams()
            params += self.mrec_ball.getParams()
        if self.isTrainingGenerativeModel:
            params += self.mprior.getParams()
        return params

    def EnableGenerativeModelTraining(self):
        '''
        Enable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = True
        print('Enable switching training/test mode in generative model class!\n')

    def DisableGenerativeModelTraining(self):
        '''
        Disable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = False
        print('Enable switching training/test mode in generative model class!\n')

    def cost(self):
        '''
        Compute a one-sample approximation the ELBO (lower bound on marginal likelihood), normalized by batch size (length of Y in first dimension).
        '''
        if not self.noX:
            qg = self.mrec_goalie.getSample()
            qb = self.mrec_ball.getSample()
            thelik = self.mprior.evaluateLogDensity(qg, qb, self.Y)
            theentropy = self.mrec_goalie.evalEntropy() + self.mrec_ball.evalEntropy()
        else:
            thelik = self.mprior.evaluateLogDensity(None, None, self.Y)
            theentropy = 0

        theentropy += self.mprior.evalEntropy()

        thecost = thelik + theentropy

        return thecost / self.Y.shape[0]


class SGVB_NN():#(Trainable):
    '''
    This class fits a model to PenaltyKick data.

    Inputs:
    gen_params       - Dictionary of parameters that define the chosen GenerativeModel
    GEN_MODEL        - A class that inhereits from the GenerativeModel abstract class
    yDim             - Integer that specifies the dimensionality of the observations

    --------------------------------------------------------------------------

    The SGVB ("Stochastic Gradient Variational Bayes") inference technique is described
    in the following publications:
    * Auto-Encoding Variational Bayes
           - Kingma, Welling (ICLR, 2014)
    * Stochastic backpropagation and approximate inference in deep generative models.
           - Rezende et al (ICML, 2014)
    * Doubly stochastic variational bayes for non-conjugate inference.
           - Titsias and Lazaro-Gredilla (ICML, 2014)
    '''
    def __init__(self,
                 gen_params_ball,  # dictionary of generative model parameters
                 gen_params_goalie,  # dictionary of generative model parameters
                 GEN_MODEL,  # class that inherits from GenerativeModel
                 yDim_ball,  # number of observation dimensions
                 yDim_goalie,  # number of observation dimensions
                 rec_params_ball, # dictionary of approximate posterior ("recognition model") parameters
                 rec_params_goalie, # dictionary of approximate posterior ("recognition model") parameters
                 REC_MODEL,
                 ntrials):

        # instantiate rng's
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)

        #---------------------------------------------------------
        ## actual model parameters
        self.X, self.Y = T.matrices('X', 'Y')   # symbolic variables for the data
        self.mode = T.ivector('mode')

        self.yDim_goalie = yDim_goalie
        self.yDim_ball = yDim_ball
        self.xDim_goalie = yDim_goalie
        self.xDim_ball = yDim_ball
        self.yDim = self.yDim_goalie + self.yDim_ball

        # instantiate our prior and recognition models
        self.mrec_goalie = REC_MODEL(rec_params_goalie, self.Y,
                                     self.xDim_goalie, self.yDim, ntrials,
                                     self.srng, self.nrng)
        self.mrec_ball = REC_MODEL(rec_params_ball, self.Y,
                                   self.xDim_ball, self.yDim, ntrials,
                                   self.srng, self.nrng)
        self.mprior_goalie = GEN_MODEL(gen_params_goalie, self.xDim_goalie,
                                       self.yDim_goalie, self.yDim, ntrials,
                                       srng=self.srng, nrng=self.nrng)
        self.mprior_ball = GEN_MODEL(gen_params_ball, self.xDim_ball,
                                     self.yDim_ball, self.yDim, ntrials,
                                     srng=self.srng, nrng=self.nrng)

        self.set_PKbias_mode(self.mode)

        self.isTrainingGenerativeModel = True
        self.isTrainingRecognitionModel = True

    def set_PKbias_mode(self, mode):
        self.mrec_goalie.set_PKbias_mode(mode)
        self.mrec_ball.set_PKbias_mode(mode)
        self.mprior_goalie.set_PKbias_mode(mode)
        self.mprior_ball.set_PKbias_mode(mode)

    def getParams(self):
        '''
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []
        if self.isTrainingRecognitionModel:
            params += self.mrec_goalie.getParams()
            params += self.mrec_ball.getParams()
        if self.isTrainingGenerativeModel:
            params += self.mprior_goalie.getParams()
            params += self.mprior_ball.getParams()
        return params

    def EnableGenerativeModelTraining(self):
        '''
        Enable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = True
        print('Enable switching training/test mode in generative model class!\n')

    def DisableGenerativeModelTraining(self):
        '''
        Disable training of GenerativeModel parameters.
        '''
        self.isTrainingGenerativeModel = False
        print('Enable switching training/test mode in generative model class!\n')

    def cost(self):
        '''
        Compute a one-sample approximation the ELBO (lower bound on marginal likelihood), normalized by batch size (length of Y in first dimension).
        '''
        qg = self.mrec_goalie.getSample()
        qb = self.mrec_ball.getSample()
        thelik = self.mprior_goalie.evaluateLogDensity(qg, self.Y)
        thelik += self.mprior_ball.evaluateLogDensity(qb, self.Y)
        theentropy = self.mrec_goalie.evalEntropy() + self.mrec_ball.evalEntropy()

        thecost = thelik + theentropy

        return thecost / self.Y.shape[0]
