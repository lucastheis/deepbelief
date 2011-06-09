import numpy as np
import utils
from abstractbm import AbstractBM
from rbm import RBM

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

class SemiRBM(AbstractBM):
	"""
	An implementation of the semi-restricted Boltzmann machine. In contrast
	to the L{RBM}, the L{SemiRBM} can also have lateral connections between
	visible units.

	B{References:}
		- Osindero, S. and Hinton, G.E. (2008). I{Modeling image patches with a
		directed hiarchy of Markov random fields.}

	@type L: matrix
	@ivar L: weight matrix connecting visible units

	@type learning_rate_lateral: real
	@ivar sampling_method: step width of gradient descent for lateral connections

	@type momentum_lateral: real
	@ivar momentum_lateral: momentum for lateral connections

	@type weight_decay_lateral: real
	@ivar weight_decay_lateral: weight decay for lateral connections
	"""

	def __init__(self, num_visibles, num_hiddens):
		"""
		Initializes the parameters of the SemiRBM.

		@type  num_visibles: integer
		@param num_visibles: number of visible units
		@type  num_hiddens:  integer
		@param num_hiddens:  number of hidden units
		"""

		AbstractBM.__init__(self, num_visibles, num_hiddens)

		# additional hyperparameters
		self.learning_rate_lateral = 0.01
		self.momentum_lateral = 0.5
		self.weight_decay_lateral = 0.

		self.damping = 0.2
		self.num_lateral_updates = 20
		self.sampling_method = AbstractBM.MF

		# additional parameters
		self.L = np.matrix(np.random.randn(num_visibles, num_visibles)) / num_visibles / 200
		self.L = np.triu(self.L) + np.triu(self.L).T - 2. * np.diag(np.diag(self.L))
		self.dL = np.zeros_like(self.L)



	def forward(self, X=None):
		if X is None:
			X = self.X
		elif not isinstance(X, np.matrix):
			X = np.matrix(X)

		self.Q = 1. / (1. + np.exp(-self.W.T * X - self.c))
		self.Y = (np.random.rand(*self.Q.shape) < self.Q).astype(self.Q.dtype)

		return self.Y.copy()



	def backward(self, Y=None, X=None):
		"""
		Conditionally samples the visible units using either sequential Gibbs
		sampling or parallel mean field updates.

		@type Y:  array_like
		@param Y: states of hidden units

		@type X:  array_like
		@param X: states of visible units

		@rtype:  matrix
		@return: a matrix containing states for the visible units
		"""

		if Y is None:
			Y = self.Y
		else:
			Y = np.asmatrix(Y)

		if X is None:
			self.X = np.asmatrix(np.zeros([self.X.shape[0], Y.shape[1]]))
		else:
			self.X = np.asmatrix(X)

		# constant input coming from the hidden units
		dynamic_bias = self.W * Y + self.b

		if self.sampling_method is AbstractBM.MF:
			self.P = 1. / (1. + np.exp(-dynamic_bias - self.L * self.X))

			# parallel mean field updates
			for k in range(self.num_lateral_updates):
				self.P = self.damping * self.P + (1. - self.damping) * 1. / (1. + np.exp(-dynamic_bias - self.L * self.P))

			self.X = (np.random.rand(*self.P.shape) < self.P).astype(self.X.dtype)

		else:
			self.P = np.zeros_like(self.X)

			# sequential Gibbs updates
			for k in range(self.num_lateral_updates + 1):
				# compute activation probabilities in random order
				for i in np.random.permutation(len(self.X)):
					self.P[i, :] = 1. / (1. + np.exp(-dynamic_bias[i, :] - self.L[i, :] * self.X))
					self.X[i, :] = np.random.rand(1, Y.shape[1]) < self.P[i, :]

		return self.X.copy()



	def train(self, X):
		X = np.asmatrix(X)

		if self.sampling_method is AbstractBM.MF:
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
			self.dW = X * Q.T / X.shape[1] - self.P * self.Y.T / self.X.shape[1] \
			        - self.weight_decay * self.W \
			        + self.momentum * self.dW
			self.db = X.mean(1) - self.P.mean(1) + self.momentum * self.db
			self.dc = Q.mean(1) - self.Y.mean(1) + self.momentum * self.dc
			self.dL = X * X.T / X.shape[1] - self.P * self.P.T / self.X.shape[1] \
			        - self.weight_decay_lateral * self.L \
			        + self.momentum_lateral * self.dL
			self.dL -= np.diag(np.diag(self.dL))

			self.W += self.dW * self.learning_rate
			self.b += self.db * self.learning_rate
			self.c += self.dc * self.learning_rate
			self.L += self.dL * self.learning_rate_lateral
		else:
			# update RBM parameters
			AbstractBM.train(self, X)

			# update lateral connections
			self.dL = X * X.T / X.shape[1] - self.X * self.X.T / self.X.shape[1] \
			        - self.weight_decay_lateral * self.L \
			        + self.momentum_lateral * self.dL
			self.dL -= np.diag(np.diag(self.dL))
			
			self.L += self.dL * self.learning_rate_lateral



	def _train_wake(self, X, Y):
		X = np.asmatrix(X)
		Y = np.asmatrix(Y)

		self.backward(Y)

		if self.sampling_method is AbstractBM.MF:
			self.db = X.mean(1) - self.P.mean(1) + self.momentum * self.db
			self.dW = (X * Y.T - self.P * Y.T) / X.shape[1] + self.momentum * self.dW
			self.dL = (X * X.T - self.P * self.P.T) / self.X.shape[1]
			self.dL -= np.diag(np.diag(self.dL))
		else:
			self.db = X.mean(1) - self.X.mean(1) + self.momentum * self.db
			self.dW = (X * Y.T - self.X * Y.T) / X.shape[1] + self.momentum * self.dW
			self.dL = (X * X.T - self.X * self.X.T) / self.X.shape[1]
			self.dL -= np.diag(np.diag(self.dL))

		self.W += self.dW * self.learning_rate
		self.b += self.db * self.learning_rate
		self.L += self.dL * self.learning_rate_lateral



	def _train_sleep(self, X, Y):
		X = np.asmatrix(X)
		Y = np.asmatrix(Y)

		Q = 1. / (1. + np.exp(-self.W.T * X - self.c))

		tmp1 = np.multiply(Y, 1 - Q)
		tmp2 = np.multiply(Y - 1, Q)

		self.dW = X * (tmp1 + tmp2).T / X.shape[1] + self.momentum * self.dW
		self.dc = tmp1.mean(1) + tmp2.mean(1) + self.momentum * self.dc

		self.W += self.dW * self.learning_rate
		self.c += self.dc * self.learning_rate



	def _ulogprob(self, X, Y, all_pairs=False):
		X = np.asmatrix(X)
		Y = np.asmatrix(Y)

		if all_pairs:
			return X.T * self.b + self.c.T * Y \
			   + np.sum(np.multiply(X, self.L * X), 0).T / 2. \
			   + X.T * self.W * Y
		else:
			return self.b.T * X + self.c.T * Y \
			   + np.sum(np.multiply(X, self.L * X), 0) / 2. \
			   + np.sum(np.multiply(X, self.W * Y), 0)



	def _ulogprob_vis(self, X):
		X = np.asmatrix(X)

		return self.b.T * X \
		   + np.sum(np.log(1. + np.exp(self.W.T * X + self.c)), 0) \
		   + np.sum(np.multiply(X, self.L * X), 0) / 2.



	def _ulogprob_hid(self, Y, num_is_samples=100):
		"""
		Estimates the unnormalized marginal log-probabilities of hidden states.
		
		Use this method only if you know what you are doing.
		"""

		# approximate this SRBM with an RBM
		rbm = RBM(self.X.shape[0], self.Y.shape[0])
		rbm.W = self.W
		rbm.b = self.b
		rbm.c = self.c

		# allocate memory
		Q = np.asmatrix(np.zeros([num_is_samples, Y.shape[1]]))

		for k in range(num_is_samples):
			# draw importance samples
			X = rbm.backward(Y)

			# store importance weights
			Q[k, :] = self._ulogprob(X, Y) - rbm._clogprob_vis_hid(X, Y)

		# average importance weights to get estimates
		return utils.logmeanexp(Q, 0)



	def _clogprob_hid_vis(self, X, Y, all_pairs=False):
		X = np.asmatrix(X)
		Y = np.asmatrix(Y)

		Q = 1. / (1. + np.exp(-self.W.T * X - self.c))

		if all_pairs:
			return np.log(Q).T * Y + np.log(1. - Q).T * (1. - Y)
		else:
			return np.sum(np.log(2 * np.multiply(Q, Y) - Y - (Q - 1)), 0)



	def _centropy_hid_vis(self, X):
		# compute probabilities of hidden units
		self.forward(X)

		# compute entropy
		return -np.sum(
		   np.multiply(self.Q, np.log(self.Q)) + \
		   np.multiply(1 - self.Q, np.log(1 - self.Q)), 0)
