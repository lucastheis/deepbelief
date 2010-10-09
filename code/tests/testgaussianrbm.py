import sys
import unittest
import numpy as np

sys.path.append('./code')

from deepbelief import GaussianRBM, utils

class TestGaussianRBM(unittest.TestCase):
	def test_probabilities(self):
		grbm = GaussianRBM(7, 13)
		grbm.W = np.asmatrix(np.random.randn(grbm.X.shape[0], grbm.Y.shape[0]))
		grbm.b = np.asmatrix(np.random.rand(grbm.X.shape[0], 1))
		grbm.c = np.asmatrix(np.random.randn(grbm.Y.shape[0], 1))
		grbm.sigma = np.random.rand() * 0.5 + 0.5
		grbm.sigma = 1.

		examples_vis = np.asmatrix(np.random.randn(grbm.X.shape[0], 1000) * 2.)
		examples_hid = np.matrix(np.random.rand(grbm.Y.shape[0], 100) < 0.5)

		states_hid = utils.binary_numbers(grbm.Y.shape[0])

		# check that conditional probabilities are normalized
		logprobs = grbm._clogprob_hid_vis(examples_vis, states_hid, all_pairs=True)
		self.assertTrue(np.all(utils.logsumexp(logprobs, 1) < 1E-8))

		# test for consistency
		logprobs1 = grbm._ulogprob(examples_vis, examples_hid, all_pairs=True)
		logprobs2 = grbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True) \
		          + grbm._ulogprob_hid(examples_hid)
		logprobs3 = grbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True) \
		          + grbm._ulogprob_vis(examples_vis).T

		self.assertTrue(np.all(np.abs(logprobs1 - logprobs2) < 1E-10))
		self.assertTrue(np.all(np.abs(logprobs1 - logprobs3) < 1E-3))



	def test_all_pairs(self):
		grbm = GaussianRBM(10, 20)
		grbm.W = np.matrix(np.random.randn(grbm.X.shape[0], grbm.Y.shape[0]))
		grbm.b = np.matrix(np.random.rand(grbm.X.shape[0], 1))
		grbm.c = np.matrix(np.random.randn(grbm.Y.shape[0], 1))
		grbm.sigma = np.random.rand() + 0.5

		examples_vis = np.matrix(np.random.randn(grbm.X.shape[0], 100))
		examples_hid = np.matrix(np.random.rand(grbm.Y.shape[0], 100) < 0.5)

		logprob1 = grbm._ulogprob(examples_vis, examples_hid)
		logprob2 = np.diag(grbm._ulogprob(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = grbm._ulogprob(examples_vis[:, 1], examples_hid)
		logprob2 = grbm._ulogprob(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = grbm._ulogprob(examples_vis, examples_hid[:, 1])
		logprob2 = grbm._ulogprob(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)
		
		logprob1 = grbm._clogprob_vis_hid(examples_vis, examples_hid)
		logprob2 = np.diag(grbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = grbm._clogprob_vis_hid(examples_vis[:, 1], examples_hid)
		logprob2 = grbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = grbm._clogprob_vis_hid(examples_vis, examples_hid[:, 1])
		logprob2 = grbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = grbm._clogprob_hid_vis(examples_vis, examples_hid)
		logprob2 = np.diag(grbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-5)

		logprob1 = grbm._clogprob_hid_vis(examples_vis[:, 1], examples_hid)
		logprob2 = grbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-5)

		logprob1 = grbm._clogprob_hid_vis(examples_vis, examples_hid[:, 1])
		logprob2 = grbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-5)




if __name__ == '__main__':
	unittest.main()
