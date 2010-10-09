import sys
import unittest
import numpy as np

sys.path.append('./code')

from deepbelief import RBM, utils

class TestRBM(unittest.TestCase):
	def test_probs(self):
		rbm = RBM(9, 11)
		rbm.W = np.matrix(np.random.randn(rbm.X.shape[0], rbm.Y.shape[0]))
		rbm.b = np.matrix(np.random.rand(rbm.X.shape[0], 1))
		rbm.c = np.matrix(np.random.randn(rbm.Y.shape[0], 1))

		examples_vis = np.matrix(np.random.rand(rbm.X.shape[0], 100) < 0.5)
		examples_hid = np.matrix(np.random.rand(rbm.Y.shape[0], 100) < 0.5)

		states_vis = utils.binary_numbers(rbm.X.shape[0])
		states_hid = utils.binary_numbers(rbm.Y.shape[0])

		# check that conditional probabilities are normalized
		logprobs = rbm._clogprob_vis_hid(states_vis, examples_hid, all_pairs=True)
		self.assertTrue(np.all(utils.logsumexp(logprobs, 0) < 1E-10))

		logprobs = rbm._clogprob_hid_vis(examples_vis, states_hid, all_pairs=True)
		self.assertTrue(np.all(utils.logsumexp(logprobs, 1) < 1E-10))

		# test for consistency
		logprobs1 = rbm._ulogprob(examples_vis, examples_hid, all_pairs=True)
		logprobs2 = rbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True) \
		          + rbm._ulogprob_hid(examples_hid)
		logprobs3 = rbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True) \
		          + rbm._ulogprob_vis(examples_vis).T
		self.assertTrue(np.all(np.abs(logprobs1 - logprobs2) < 1E-10))
		self.assertTrue(np.all(np.abs(logprobs1 - logprobs3) < 1E-10))




	def test_all_pairs(self):
		rbm = RBM(10, 20)
		rbm.W = np.matrix(np.random.randn(rbm.X.shape[0], rbm.Y.shape[0]))
		rbm.b = np.matrix(np.random.rand(rbm.X.shape[0], 1))
		rbm.c = np.matrix(np.random.randn(rbm.Y.shape[0], 1))

		examples_vis = np.matrix(np.random.rand(rbm.X.shape[0], 100) < 0.5)
		examples_hid = np.matrix(np.random.rand(rbm.Y.shape[0], 100) < 0.5)

		logprob1 = rbm._ulogprob(examples_vis, examples_hid)
		logprob2 = np.diag(rbm._ulogprob(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = rbm._ulogprob(examples_vis[:, 1], examples_hid)
		logprob2 = rbm._ulogprob(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = rbm._ulogprob(examples_vis, examples_hid[:, 1])
		logprob2 = rbm._ulogprob(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)
		
		logprob1 = rbm._clogprob_vis_hid(examples_vis, examples_hid)
		logprob2 = np.diag(rbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = rbm._clogprob_vis_hid(examples_vis[:, 1], examples_hid)
		logprob2 = rbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = rbm._clogprob_vis_hid(examples_vis, examples_hid[:, 1])
		logprob2 = rbm._clogprob_vis_hid(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = rbm._clogprob_hid_vis(examples_vis, examples_hid)
		logprob2 = np.diag(rbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = rbm._clogprob_hid_vis(examples_vis[:, 1], examples_hid)
		logprob2 = rbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = rbm._clogprob_hid_vis(examples_vis, examples_hid[:, 1])
		logprob2 = rbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)



if __name__ == '__main__':
	unittest.main()
