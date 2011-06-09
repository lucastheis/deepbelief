import numpy as np
import utils
from abstractbm import AbstractBM
from mixbm import MixBM
from basebm import BaseBM
from semirbm import SemiRBM
from dbn import DBN
from tools.parallel import map
from tools import shmarray

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

class Estimator:
	"""
	This class implements annealed importance sampling for the estimation of
	partition functions and means to estimate log-probabilities of data
	samples.

	Always estimate the partition function of a deep belief network first.

		>>> dbn.estimate_log_partition_function(self, num_ais_samples=100, beta_weights=np.arange(0, 1, 1000))

	The choice of the parameters is crucial. More samples and weights will lead
	to less biased estimates. Only after the partition function has been
	estimated with appropriate parameters should L{estimate_log_probability} be
	called.

		>>> logprob, lowerbound = dbn.estimate_log_probability(data, num_samples=100)

	Taking more samples will reduce the variance of the estimates.

	If one of the lower layers is an instance of L{SemiRBM}, then
	L{estimate_log_partition_function} also has to be run for this layer with
	a I{large} value for C{num_ais_samples}.

	B{References:}
		- Salakhutdinov, R. and Murray, I. (2008). I{On the Quantitative Analysis
		of Deep Belief Networks.}
		- Theis, L., Gerwinn, S., Sinz, F. Bethge, M. (2010). I{Likelihood
		Estimation in Deep Belief Networks.}
	"""

	def __init__(self, dbn):
		"""
		Prepare the sampler.
		"""

		if not isinstance(dbn, DBN):
			if isinstance(dbn, AbstractBM):
				dbn = DBN(dbn)
			else:
				raise TypeError('DBN or RBM expected.')

		self.dbn = dbn

		for l in range(len(self.dbn)):
			if not hasattr(self.dbn[l], '_ais_logz'):
				self.dbn[l]._ais_logz = None
				self.dbn[l]._ais_samples = None
				self.dbn[l]._ais_logweights = None



	def estimate_log_partition_function(self, num_ais_samples=100, beta_weights=[], layer=-1):
		"""
		Estimate the log of the partition function.

		C{beta_weights} should be a list of monotonically increasing values ranging
		from 0 to 1. See Salakhutdinov & Murray (2008) for details on how to set
		the parameters.

		@type  num_ais_samples: integer
		@param num_ais_samples: number of samples used to estimate the partition
		function

		@type  beta_weights: array_like
		@param beta_weights: annealing weights ranging from zero to one

		@type  layer: integer
		@param layer: can be used to estimate the partition function of one
		of the lower layers

		@rtype:  real
		@return: the estimated log partition function
		"""

		bsbm = BaseBM(self.dbn[layer])
		mxbm = MixBM(bsbm, self.dbn[layer])

		# settings relevant only for SemiRBM
		bsbm.sampling_method = AbstractBM.GIBBS
		mxbm.sampling_method = AbstractBM.GIBBS
		mxbm.num_lateral_updates = 5

		# draw (independent) samples from the base model
		X = bsbm.sample(num_ais_samples, 0, 1)

		# compute importance weights
		logweights = bsbm._free_energy(X)

		for beta in beta_weights:
			mxbm.tune(beta)

			logweights -= mxbm._free_energy(X)
			Y = mxbm.forward(X)
			X = mxbm.backward(Y, X)
			logweights += mxbm._free_energy(X)

		logweights -= self.dbn[layer]._free_energy(X)

		# store results for later use
		self.dbn[layer]._ais_logweights = logweights + bsbm.logz
		self.dbn[layer]._ais_logz = utils.logmeanexp(logweights) + bsbm.logz
		self.dbn[layer]._ais_samples = X

		return self.dbn[layer]._ais_logz



	def estimate_log_probability(self, X, num_samples=200):
		"""
		Estimates the log-probability in nats.
		
		This method returns two values: Optimistic but consistent estimates of
		the log probability of the given data samples and estimated lower bounds.
		The parameter C{num_samples} is only relevant for DBNs with at least 2
		layers.  L{estimate_log_partition_function}() should be run with
		appropriate parameters beforehand, otherwise the probability estimates
		will be very poor.

		@type  X: array_like
		@param X: the data points for which to estimate the log-probability

		@type  num_samples: integer
		@param num_samples: the number of Monte Carlo samples used to estimate the
		unnormalized probability of the data samples

		@rtype:  tuple
		@return: a tuple consisting of the estimated log-probabilities (first entry)
		and estimated lower bounds (second entry)
		"""

		# estimate partition function if not done yet
		if not self.dbn[-1]._ais_logz:
			self.estimate_log_partition_function()

		if len(self.dbn) > 1:
			for l in range(len(self.dbn) - 1):
				if isinstance(self.dbn[l], SemiRBM):
					# needed for estimating SemiRBM marginals
					if not self.dbn[l]._ais_logz:
						self.dbn[l]._ais_logz = self.estimate_log_partition_function(layer=l)

			# allocate (shared) memory for log importance weights
			logiws = shmarray.zeros([num_samples, X.shape[1]])

			# Monte Carlo estimation of unnormalized probability
			def parfor(i):
				samples = X

				for l in range(len(self.dbn) - 1):
					logiws[i, :] += self.dbn[l]._ulogprob_vis(samples).A[0]
					samples = self.dbn[l].forward(samples)
					logiws[i, :] -= self.dbn[l]._ulogprob_hid(samples).A[0]
				logiws[i, :] += self.dbn[-1]._ulogprob_vis(samples).A[0]
			map(parfor, range(num_samples))

			# averaging weights yields unnormalized probability
			ulogprob = utils.logmeanexp(np.asmatrix(logiws), 0)
			ubound = logiws.mean(0)

		else:
			ulogprob = self.dbn[0]._ulogprob_vis(X)
			ubound = ulogprob.copy()

		# return normalized log probability
		return (ulogprob - self.dbn[-1]._ais_logz, ubound - self.dbn[-1]._ais_logz)
