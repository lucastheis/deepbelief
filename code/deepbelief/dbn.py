import numpy as np
import utils
import copy
from semirbm import SemiRBM

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

class DBN:
	"""
	This class allows the construction of simpler deep belief network
	architectures. It facilitates training and sampling from I{stacks} of
	Boltzmann machines. However, this class does currently not support tree
	structures or other more complicated arrangements.

	Only after the current top most layer has been trained using L{train}() should a
	new layer be added to the DBN using L{add_layer}(). After adding a new layer, the
	lower layers can no longer be trained.

	B{Example:}

	Create and train a first layer.

			>>> dbn = DBN(RBM(10, 20))
			>>> dbn[0].learning_rate = 1E-3
			>>> dbn[0].momentum = 0.8
			>>> dbn[0].weight_decay = 1E-2
			>>> dbn.train(data, num_epochs=50)

	Afterwards, add and train a second layer.

			>>> dbn.add_layer(RBM(20, 50))
			>>> dbn[1].learning_rate = 1E-3
			>>> dbn[1].momentum = 0.8
			>>> dbn[1].weight_decay = 1E-2
			>>> dbn.train(data, num_epochs=50)

	Note that each layer has its own set of training parameters.

	B{References:}
		- Hinton, G. E. and Salakhutdinov, R. (2006). I{Reducing the Dimensionality
		of Data with Neural Networks.} Science.
		- Hinton, G.E., P. Dayan, B. J. Frey and R. M. Neal (1995). I{The "wake-sleep" algorithm
		for unsupervised neural networks.} Science.
	"""

	def __init__(self, model):
		"""
		Initializes a deep belief network with one layer.

		@type  model: AbstractBM
		@param model: first layer of the deep belief network
		"""

		self.models = []
		self.models.append(model)

		# the proposal DBN used during wake-sleep training
		self.proposal = None



	def __getitem__(self, key):
		"""
		Returns the model at the specified position in the hierarchy.

		@rtype: AbstractBM
		"""

		return self.models[key]



	def __len__(self):
		"""
		Returns the number of layers.

		@rtype: integer
		"""
		return len(self.models)



	def forward(self, X):
		"""
		Passes some input through the network and returns a sample for the top
		hidden units.

		@type  X: array_like
		@param X: states of visible units
		@rtype:   matrix
		@return:  a matrix containing states for the top hidden units
		"""

		for model in self.models:
			X = model.forward(X)

		return X



	def backward(self, Y):
		"""
		Passes a state from the top hidden units back to the visible units.

		@type Y:  array_like
		@param Y: states of hidden units

		@rtype:  matrix
		@return: a matrix containing states for the visible units
		"""

		for model in reversed(self.models):
			Y = model.backward(Y)

		return Y



	def add_layer(self, model):
		"""
		Adds a new layer to the deep belief network.

		@type  model: AbstractBM
		@param model: Boltzmann machine which will be appended to the network

		@raise ValueError: raised if model is incompatible to the current top layer
		"""

		if len(self.models[-1].Y) != len(model.X):
			raise ValueError('The number of visible units of the new layer must be {0}.'.format(len(self.models[-1].Y)))

		self.models.append(model)



	def sample(self, num_samples=1, burn_in_length=100, sample_spacing=20, num_parallel_chains=1):
		"""
		Draws samples from the model.

		@type  num_samples: integer
		@param num_samples: the number of samples to draw from the model

		@type  burn_in_length: integer
		@param burn_in_length: the number of discarded initial samples

		@type  sample_spacing: integer
		@param sample_spacing: return only every I{n}-th sample of the Markov chain

		@type  num_parallel_chains: integer
		@param num_parallel_chains: number of parallel Markov chains

		@rtype:  matrix
		@return: a matrix containing the drawn samples in its columns
		"""

		# sample from the top model
		Y = self.models[-1].sample(num_samples, burn_in_length, sample_spacing, num_parallel_chains)

		# propagate samples down to the visible units
		for model in reversed(self.models[:-1]):
			Y = model.backward(Y)

		return Y



	def estimate_log_partition_function(self, num_ais_samples=100, beta_weights=[], layer=-1):
		"""
		Estimate the log of the partition function of the top layer. This method is
		a wrapper for the L{Estimator} class and is provided for convenience.

		@type  num_ais_samples: integer
		@param num_ais_samples: number of samples used to estimate the partition function

		@type  beta_weights: list
		@param beta_weights: annealing weights changing from zero to one

		@rtype:  real
		@return: log of the estimated partition function
		"""

		import estimator
		return estimator.Estimator(self).estimate_log_partition_function(num_ais_samples, beta_weights, layer)



	def estimate_log_likelihood(self, X, num_samples=200):
		"""
		Estimate the log probability of a set of data samples. This method uses
		the L{Estimator} class.

		@type  X: array_like
		@param X: data points

		@type  num_samples: integer
		@param num_samples: number of Monte Carlo samples used to estimate unnormalized
		probabilities

		@rtype:  real
		@return: the average model log-likelihood in nats
		"""

		import estimator
		return estimator.Estimator(self).estimate_log_probability(X, num_samples)[0].mean()



	def train(self, X, num_epochs=50, batch_size=0, shuffle=True, learning_rates=None):
		"""
		This method greedily trains the top level of the DBN while keeping the
		other models fixed.

		If C{batch_size} is 0, the model is trained on all samples at once.
		Otherwise, the training samples are split into batches and several
		updates are performed per epoch.  If C{shuffle} is set to true, then the
		training samples are randomly shuffled before each new epoch.  If
		C{learning_rates} is an array or a list with num_epochs entries, the
		learning rate at the I{i}-th iteration is set to its I{i}-th entry. If
		C{learning_rates} has 2 entries, the learning rate is linearly annealed
		over all epochs from the first to the second entry.

		B{Example}:

			>>> dbn.train(data, 50, 100, True, [1E-2, 1E-4])

		@type  num_epochs: integer
		@param num_epochs: number of iterations of the algorithm

		@type  batch_size: integer
		@param batch_size: size of data batches used for training

		@type  shuffle: boolean
		@param shuffle: randomize order of data before each iteration

		@type  learning_rates: array_like
		@param learning_rates: different learning rates for each epoch
		"""

		if not batch_size:
			batch_size = X.shape[1]

		if isinstance(learning_rates, float):
			learning_rates = np.linspace(learning_rates, learning_rates, num_epochs)
		elif not learning_rates is None and len(learning_rates) != num_epochs:
			learning_rates = np.linspace(learning_rates[0], learning_rates[-1], num_epochs)

		for epoch in range(num_epochs):
			# adapt learning rate
			if not learning_rates is None:
				self.models[-1].learning_rate = learning_rates[epoch]
				self.models[-1].learning_rate_lateral = learning_rates[epoch]

			# feed input through the net
			Y = X
			for model in self.models[:-1]:
				Y = model.forward(Y)

			if shuffle:
				# randomize order of data points
				Y = Y[:, np.random.permutation(Y.shape[1])]

			# train model on batches of training data
			for i in range(0, Y.shape[1], batch_size):
				self.models[-1].train(Y[:, i:i + batch_size])

		# reset models to free memory
		for model in self.models:
			model.clear()



	def train_wake_sleep(self, X, num_epochs=50, batch_size=0, shuffle=True):
		"""
		An implementation of the wake-sleep algorithm for training DBNs.

		@type  X: array_like
		@param X: data points stored in columns

		@type  num_epochs: integer
		@param num_epochs: number of iterations of the algorithm

		@type  batch_size: integer
		@param batch_size: size of data batches used for training

		@type  shuffle: boolean
		@param shuffle: randomize order of data before each iteration
		"""

		for model in self.models:
			# make sure persistent Markov chains are used during sleep phase
			model.persistent = True

			# reset gradients
			model.dW *= 0.
			model.db *= 0.
			model.dc *= 0.

			if model.__class__ is SemiRBM:
				model.dL *= 0.

		if not batch_size:
			batch_size = X.shape[1]

		if self.proposal is None:
			self.proposal = copy.deepcopy(self)

		for epoch in range(num_epochs):
			if shuffle:
				# randomize order of data points
				X = X[:, np.random.permutation(X.shape[1])]

			# train on batches
			for i in range(0, X.shape[1], batch_size):
				Y = [X[:, i:i + batch_size]]

				# wake phase (train the model)
				for j in range(len(self) - 1):
					Y.append(self.proposal[j].forward(Y[-1]))
					self.models[j]._train_wake(Y[-2], Y[-1])
				self.models[-1].train(Y[-1])

				# sleep phase (adapt the proposal distribution)
				Y = [self.models[-1].sample(num_samples=batch_size,
					num_parallel_chains=batch_size, burn_in_length=1, sample_spacing=1)]

				for j in range(len(self) - 2, -1, -1):
					Y.append(self.models[j].backward(Y[-1]))
					self.proposal[j]._train_sleep(Y[-1], Y[-2])
