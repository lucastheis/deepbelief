import numpy as np
from abstractbm import AbstractBM

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

class RBM(AbstractBM):
	"""
	An implementation of the restricted Boltzmann machine.
	
	B{References:}
		- Hinton, G. E. (2002). I{Training Products of Experts by Minimizing
		Contrastive Divergence.} Neural Computation.
	"""

	def __init__(self, num_visibles, num_hiddens):
		"""
		Initializes the parameters of the RBM.

		@type  num_visibles: integer
		@param num_visibles: number of visible units
		@type  num_hiddens:  integer
		@param num_hiddens:  number of hidden units
		"""

		AbstractBM.__init__(self, num_visibles, num_hiddens)



	def forward(self, X=None):
		if X is None:
			X = self.X
		elif not isinstance(X, np.matrix):
			X = np.asmatrix(X)
		
		self.Q = 1. / (1. + np.exp(-self.W.T * X - self.c))
		self.Y = (np.random.rand(*self.Q.shape) < self.Q).astype(self.Q.dtype)

		return self.Y.copy()



	def backward(self, Y=None, X=None):
		if Y is None:
			Y = self.Y
		else:
			Y = np.asmatrix(Y)

		self.P = 1. / (1. + np.exp(-self.W * Y - self.b))
		self.X = (np.random.rand(*self.P.shape) < self.P).astype(self.P.dtype)

		return self.X.copy()



	def _train_wake(self, X, Y):
		X = np.asmatrix(X)
		Y = np.asmatrix(Y)

		P = 1. / (1. + np.exp(-self.W * Y - self.b))

		tmp1 = np.multiply(X, 1 - P)
		tmp2 = np.multiply(X - 1, P)

		self.dW = (tmp1 + tmp2) * Y.T / X.shape[1] + self.momentum * self.dW
		self.db = tmp1.mean(1) + tmp2.mean(1) + self.momentum * self.db

		self.W += self.dW * self.learning_rate
		self.b += self.db * self.learning_rate



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
			return X.T * self.W * Y + X.T * self.b + self.c.T * Y
		else:
			return np.sum(np.multiply(X, self.W * Y), 0) + self.b.T * X + self.c.T * Y



	def _ulogprob_vis(self, X):
		return self.b.T * X + np.sum(np.log(1. + np.exp(self.W.T * X + self.c)), 0)



	def _ulogprob_hid(self, Y):
		return self.c.T * Y + np.sum(np.log(1. + np.exp(self.W * Y + self.b)), 0)



	def _clogprob_vis_hid(self, X, Y, all_pairs=False):
		X = np.asmatrix(X)
		Y = np.asmatrix(Y)

		P = 1. / (1. + np.exp(-self.W * Y - self.b))

		if all_pairs:
			return X.T * np.log(P) + (1. - X).T * np.log(1. - P)
		else:
			return np.sum(np.log(np.multiply(P, X) + np.multiply(1 - P, 1 - X)), 0)



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
