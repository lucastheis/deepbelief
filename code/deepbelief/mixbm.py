import numpy as np
from rbm import RBM
from gaussianrbm import GaussianRBM
from semirbm import SemiRBM

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

class MixBM(RBM, GaussianRBM, SemiRBM):
	"""
	This class can be used to draw samples from a Boltzmann machine whose energy
	is a weighted sum of the energies of two given L{RBMs<RBM>}, L{SemiRBMs<SemiRBM>}
	or L{GaussianRBMs<GaussianRBM>}.

	B{References:}
		- Salakhutdinov, R. and Murray, I. (2008). I{On the Quantitative Analysis of
		Deep Belief Networks.}
	"""

	def __init__(self, model_one, model_two, weight=0.):
		"""
		Initializes the parameters of the mixed model.

		@type  model_one: AbstractBM
		@param model_one: first model

		@type  model_two: AbstractBM
		@param model_two: second model
		"""
		
		if not model_one.__class__ is model_two.__class__:
			if model_one.__class__ is SemiRBM and model_two.__class__ is RBM:
				model_two.L = 0
			elif model_one.__class__ is RBM and model_two.__class__ is SemiRBM:
				model_one.L = 0
			else:
				raise TypeError("The two models must have compatible types.")

		if len(model_one.X) != len(model_two.X):
			raise ValueError("The two models must have the same number of visible units.")

		self.rbm_type = model_two.__class__
		self.rbm_type.__init__(self, len(model_one.X), len(model_one.Y) + len(model_two.Y))

		self.sampling_method = model_two.sampling_method
		self.weight = weight
		self.model_one = model_one
		self.model_two = model_two
		self.tune(weight)



	def tune(self, weight):
		"""
		Tunes the parameters of the mixed model so that its energy becomes

			- I{(1 - weight) * energy_one + weight * energy_two.}

		@type  weight: real
		@param weight: a weight between 0 and 1
		"""

		self.weight = weight

		one = self.model_one
		two = self.model_two

		n = len(one.Y)

		if self.rbm_type == RBM:
			# adjust parameters of the mixed RBM
			self.b = (1. - weight) * one.b + weight * two.b
			self.c[:n] = one.c * (1. - weight)
			self.c[n:] = two.c * weight
			self.W[:, :n] = one.W * (1. - weight)
			self.W[:, n:] = two.W * weight

		elif self.rbm_type == GaussianRBM:
			tmp_one = 1 / (one.sigma * one.sigma) * (1 - weight)
			tmp_two = 1 / (two.sigma * two.sigma) * weight

			# adjust parameters of the mixed GaussianRBM
			self.sigma = 1 / np.sqrt(tmp_one + tmp_two)
			self.b = (tmp_one * one.b + tmp_two * two.b) * self.sigma * self.sigma
			self.W[:, :n] = one.W / one.sigma * self.sigma * (1 - weight)
			self.W[:, n:] = two.W / two.sigma * self.sigma * weight
			self.c[:n] = one.c * (1 - weight)
			self.c[n:] = two.c * weight

		elif self.rbm_type == SemiRBM:
			# adjust parameters of the mixed SemiRBM
			self.L = (1. - weight) * one.L + weight * two.L
			self.b = (1. - weight) * one.b + weight * two.b
			self.c[:n] = one.c * (1. - weight)
			self.c[n:] = two.c * weight
			self.W[:, :n] = one.W * (1. - weight)
			self.W[:, n:] = two.W * weight



	def forward(self, Y=None):
		return self.rbm_type.forward(self, Y)

	def backward(self, Y=None, X=None):
		return self.rbm_type.backward(self, Y, X)

	def train(self, X):
		self.rbm_type.train(self, X)

	def sample(self, num_samples=1, burn_in_length=100, sample_spacing=20, num_parallel_chains=1, X=None):
		return self.rbm_type._sample(self, num_samples, burn_in_length, sample_spacing, num_parallel_chains, X)

	def _free_energy(self, X):
		return self.rbm_type._free_energy(self, X)

	def _free_energy_gradient(self, X):
		return self.rbm_type._free_energy_gradient(self, X)

	def _ulogprob(self, X, Y, all_pairs=False):
		return self.rbm_type._ulogprob(self, X, Y, all_pairs)

	def _ulogprob_vis(self, X):
		return self.rbm_type._ulogprob_vis(self, X)

	def _ulogprob_hid(self, Y):
		return self.rbm_type._ulogprob_hid(self, Y)

	def _clogprob_vis_hid(self, X, Y, all_pairs=False):
		return self.rbm_type._clogprob_vis_hid(self, X, Y, all_pairs)

	def _clogprob_hid_vis(self, X, Y, all_pairs=False):
		return self.rbm_type._clogprob_hid_vis(self, X, Y, all_pairs)

	def _centropy_hid_vis(self, X):
		return self.rbm_type._centropy_hid_vis(self, X)
