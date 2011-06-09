import numpy as np

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

class AbstractBM:
	"""
	Provides an interface and common functionality for latent-variable
	Boltzmann machines, such as contrastive divergence learning, Gibbs
	sampling and hybrid Monte Carlo sampling.

	B{References:}
		- Hinton, G. E. (2002). I{Training Products of Experts by Minimizing
		Contrastive Divergence.} Neural Computation.
		- Neal, R. (1996). I{Bayesian Learning for Neural Networks.} Springer Verlag.

	@type X: matrix
	@ivar X: states of the visible units

	@type Y: matrix
	@ivar Y: states of the hidden units

	@type W: matrix
	@ivar W: weight matrix connecting visible and hidden units

	@type b: matrix
	@ivar b: visible biases

	@type c: matrix
	@ivar c: hidden biases

	@type learning_rate: real
	@ivar learning_rate: step width of gradient descent learning algorithm

	@type momentum: real
	@ivar momentum: parameter of the learning algorithm

	@type weight_decay: real
	@ivar weight_decay: prevents the weights from becoming too large

	@type sparseness: real
	@ivar sparseness: encourage sparse activation of the hidden units by
	modifying the biases

	@type sparseness_target: real
	@ivar sparseness_target: targeted level of activity

	@type cd_steps: integer
	@ivar cd_steps: number of Gibbs updates to approximate learning gradient

	@type persistent: boolean
	@ivar persistent: use persistent Markov chains to approximate learning gradient

	@type sampling_method: integer
	@ivar sampling_method: method for drawing samples (typically L{GIBBS})

	@type lf_steps: integer
	@ivar lf_steps: number of I{leapfrog} steps in L{HMC} sampling

	@type lf_step_size: real
	@ivar lf_step_size: size of one leapfrog step

	@type lf_adaptive: boolean
	@ivar lf_adaptive: automatically adjust C{lf_step_size}
	"""

	# sampling type constants
	GIBBS, HMC, MF = range(3)

	def __init__(self, num_visibles, num_hiddens):
		"""
		Initializes common parameters of Boltzmann machines.

		@type  num_visibles: integer
		@param num_visibles: number of visible units
		@type  num_hiddens:  integer
		@param num_hiddens:  number of hidden units
		"""

		# hyperparameters
		self.learning_rate = 0.01
		self.weight_decay = 0.001
		self.momentum = 0.5

		self.cd_steps = 1
		self.persistent = False

		self.sparseness = 0.0
		self.sparseness_target = 0.1

		self.sampling_method = AbstractBM.GIBBS

		# relevant for HMC sampling
		self.lf_steps = 10
		self.lf_step_size = 0.01
		self.lf_adaptive = True

		# parameters
		self.W = np.asmatrix(np.random.randn(num_visibles, num_hiddens)) / (num_visibles + num_hiddens)
		self.b = np.asmatrix(np.zeros(num_visibles)).T
		self.c = np.asmatrix(np.zeros(num_hiddens) - 1.).T

		# increments
		self.dW = np.zeros_like(self.W)
		self.db = np.zeros_like(self.b)
		self.dc = np.zeros_like(self.c)

		# variables
		self.X = np.asmatrix(np.zeros(num_visibles)).T
		self.Y = np.asmatrix(np.zeros(num_hiddens)).T

		# probabilities
		self.P = np.zeros_like(self.X)
		self.Q = np.zeros_like(self.Y)

		# states of persistent Markov chain
		self.pX = np.zeros([num_visibles, 100])
		self.pY = np.zeros([num_hiddens, 100])

		# used by annealed importance sampling
		self.ais_logz = None
		self.ais_samples = None
		self.ais_logweights = None



	def forward(self, X=None):
		"""
		Conditionally samples the hidden units. If no input is given, the current
		state of the visible units is used.

		@type  X: array_like
		@param X: states of visible units
		@rtype:   matrix
		@return:  a matrix containing states for the hidden units
		"""

		raise Exception('Abstract method \'forward\' not implemented in ' + str(self.__class__))



	def backward(self, Y=None, X=None):
		"""
		Conditionally samples the visible units. If C{Y} or C{X} is given, the
		state of the Boltzmann machine is changed prior to sampling.

		@type Y:  array_like
		@param Y: states of hidden units

		@type X:  array_like
		@param X: states of visible units

		@rtype:  matrix
		@return: a matrix containing states for the visible units
		"""

		raise Exception('Abstract method \'backward\' not implemented in ' + str(self.__class__))



	def sample(self, num_samples=1, burn_in_length=100, sample_spacing=20, num_parallel_chains=1, X=None):
		"""
		Draws samples from the model using Gibbs or hybrid Monte Carlo sampling.

		@type  num_samples: integer
		@param num_samples: the number of samples to draw from the model

		@type  burn_in_length: integer
		@param burn_in_length: the number of discarded initial samples

		@type  sample_spacing: integer
		@param sample_spacing: return only every I{n}-th sample of the Markov chain

		@type  num_parallel_chains: integer
		@param num_parallel_chains: number of parallel Markov chains

		@type  X: array_like
		@param X: initial state(s) of Markov chain(s)

		@rtype:  matrix
		@return: a matrix containing the drawn samples in its columns
		"""

		# preparations
		if self.persistent and self.pX.shape[1] == num_parallel_chains:
			self.X = self.pX
		else:
			self.X = np.zeros([self.X.shape[0], num_parallel_chains]) if X is None else X
			self.X = np.asmatrix(self.X)

		sample_step = self._sample_hmc_step if self.sampling_method is AbstractBM.HMC else self._sample_gibbs_step
		samples = []

		# burn-in phase
		for t in range(burn_in_length - sample_spacing):
			sample_step()

		for s in range(int(np.ceil(num_samples / float(num_parallel_chains)))):
			for t in range(sample_spacing):
				sample_step()
			samples.append(self.X.copy())

		if self.persistent:
			self.pX = self.X.copy()
			self.pY = self.Y.copy()

		return np.concatenate(samples, 1)[:, :num_samples]



	def train(self, X):
		"""
		Trains the parameters of the BM on a batch of data samples. The
		data stored in C{X} is used to estimate the likelihood gradient and
		one step of gradient ascend is performed.

		@type  X: array_like
		@param X: example states of the visible units
		"""

		X = np.asmatrix(X)

		# positive phase
		self.forward(X)

		# store posterior probabilities
		Q = self.Q.copy()

		if self.persistent:
			self.X = self.pX
			self.Y = self.pY

		# negative phase
		for t in range(self.cd_steps):
			self.backward()
			self.forward()

		if self.persistent:
			self.pX = self.X.copy()
			self.pY = self.Y.copy()

		# update parameters
		self.dW = X * Q.T / X.shape[1] - self.X * self.Q.T / self.X.shape[1] \
		        - self.weight_decay * self.W \
		        + self.momentum * self.dW
		self.db = X.mean(1) - self.X.mean(1) + self.momentum * self.db
		self.dc = Q.mean(1) - self.Q.mean(1) + self.momentum * self.dc
#		        - self.sparseness * np.multiply(np.multiply(Q, 1. - Q).mean(1), (Q.mean(1) - self.sparseness_target))

		self.W += self.dW * self.learning_rate
		self.b += self.db * self.learning_rate
		self.c += self.dc * self.learning_rate



	def estimate_log_partition_function(self, num_ais_samples=100, beta_weights=[], layer=-1):
		"""
		Estimate the logarithm of the partition function using annealed importance sampling.
		This method is a wrapper for the L{Estimator} class and is provided for convenience.

		@type  num_ais_samples: integer
		@param num_ais_samples: number of samples used to estimate the partition function

		@type  beta_weights: list
		@param beta_weights: annealing weights ranging from zero to one

		@rtype:  real
		@return: log of the estimated partition function
		"""

		import estimator
		return estimator.Estimator(self).estimate_log_partition_function(num_ais_samples, beta_weights, layer)



	def estimate_log_likelihood(self, X):
		"""
		Estimate the log-likelihood of the model with respect to a set of data samples.
		This method uses the L{Estimator} class.

		@type  X: array_like
		@param X: data points

		@rtype:  real
		@return: the average model log-likelihood in nats
		"""

		import estimator
		return estimator.Estimator(self).estimate_log_probability(X)[0].mean()



	def _sample_gibbs_step(self):
		"""
		Performs one step of Gibbs sampling.
		"""

		self.forward()
		self.backward()



	def _train_sleep(self, X, Y):
		"""
		Optimize conditinal likelihood for Y given X.

		@type  X: array_like
		@param X: visible states stored in columns

		@type  Y: array_like
		@param Y: hidden states stored in columns
		"""

		raise Exception('Abstract method \'_train_sleep\' not implemented in ' + str(self.__class__))



	def _train_wake(self, X, Y):
		"""
		Optimize conditinal likelihood for X given Y.

		@type  X: array_like
		@param X: visible states stored in columns

		@type  Y: array_like
		@param Y: hidden states stored in columns
		"""

		raise Exception('Abstract method \'_train_wake\' not implemented in ' + str(self.__class__))



	def clear(self):
		"""
		Reset variables. This method can help to free memory.
		"""

		# variables
		self.X = np.asmatrix(np.zeros(self.X.shape[0])).T
		self.Y = np.asmatrix(np.zeros(self.Y.shape[0])).T

		# increments
		self.dW = np.zeros_like(self.W)
		self.db = np.zeros_like(self.b)
		self.dc = np.zeros_like(self.c)

		# probabilities
		self.P = np.zeros_like(self.X)
		self.Q = np.zeros_like(self.Y)



	def _free_energy(self, X):
		return -self._ulogprob_vis(X)

	def _free_energy_gradient(self, X):
		raise Exception('Abstract method \'_free_energy_gradient\' not implemented in ' + str(self.__class__))

	def _ulogprob(self, X, Y, all_pairs=False):
		raise Exception('Abstract method \'_ulogprob\' not implemented in ' + str(self.__class__))

	def _ulogprob_vis(self, X):
		raise Exception('Abstract method \'_ulogprob_vis\' not implemented in ' + str(self.__class__))

	def _ulogprob_hid(self, Y):
		raise Exception('Abstract method \'_ulogprob_hid\' not implemented in ' + str(self.__class__))

	def _clogprob_vis_hid(self, X, Y, all_pairs=False):
		raise Exception('Abstract method \'_clogprob_vis_hid\' not implemented in ' + str(self.__class__))

	def _clogprob_hid_vis(self, X, Y, all_pairs=False):
		raise Exception('Abstract method \'_clogprob_hid_vis\' not implemented in ' + str(self.__class__))

	def _centropy_hid_vis(self, X):
		raise Exception('Abstract method \'_centropy\' not implemented in ' + str(self.__class__))
