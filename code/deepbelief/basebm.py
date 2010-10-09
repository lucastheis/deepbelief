import numpy as np
from rbm import RBM
from gaussianrbm import GaussianRBM
from semirbm import SemiRBM

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

class BaseBM(RBM, GaussianRBM, SemiRBM):
	"""
	A helper class that takes a Boltzmann machine and fits a simpler model to it.
	The partition function of the simpler model is then computed.
	An instance of a C{BaseBM} can act as a L{RBM}, L{GaussianRBM} or L{SemiRBM}.
	"""

	def __init__(self, model, num_samples=1000):
		"""
		Takes a Boltzmann machine and fits a simpler model to it.

		@type  model: AbstractBM
		@param model: an L{RBM}, L{GaussianRBM} or L{SemiRBM}

		@type  num_samples: integer
		@param num_samples: number of samples used to estimate the mean activity
		of the given model's visible units
		"""

		samples = model.sample(num_samples, 100, 20, 50)
		sample_mean = samples.mean(1)

		if model.__class__ is SemiRBM:
			self.__class__ = RBM
			self.__class__.__init__(self, model.X.shape[0], 0)
		else:
			self.__class__ = model.__class__
			self.__class__.__init__(self, model.X.shape[0], 0)

		if model.__class__ == GaussianRBM:
			self.b = sample_mean
			self.sigma = np.sqrt(np.square(samples - sample_mean).mean())
			self.logz = self.X.shape[0] / 2. * np.log(2. * np.pi) + self.X.shape[0] * np.log(self.sigma)
		else:
			sample_mean = np.minimum(np.maximum(0.1, sample_mean), 0.9)
			self.b = -np.log(1. / sample_mean - 1.)
			self.logz = np.sum(np.log(1. + np.exp(self.b)))
