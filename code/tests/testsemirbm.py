import sys
import unittest
import numpy as np

sys.path.append('./code')

from deepbelief import SemiRBM, RBM, utils

class TestSemiRBM(unittest.TestCase):
	def test_probabilities(self):
		srbm = SemiRBM(9, 12)
		srbm.W = np.matrix(np.random.randn(srbm.X.shape[0], srbm.Y.shape[0]))
		srbm.b = np.matrix(np.random.rand(srbm.X.shape[0], 1))
		srbm.c = np.matrix(np.random.randn(srbm.Y.shape[0], 1))
		srbm.L = np.matrix(np.random.randn(srbm.X.shape[0], srbm.X.shape[0])) / 2.
		srbm.L = np.triu(srbm.L) + np.triu(srbm.L).T - 2. * np.diag(np.diag(srbm.L))

		examples_vis = np.matrix(np.random.rand(srbm.X.shape[0], 100) < 0.5)
		examples_hid = np.matrix(np.random.rand(srbm.Y.shape[0], 100) < 0.5)

		states_vis = utils.binary_numbers(srbm.X.shape[0])
		states_hid = utils.binary_numbers(srbm.Y.shape[0])

		# check that conditional probabilities are normalized
		logprobs = srbm._clogprob_hid_vis(examples_vis, states_hid, all_pairs=True)
		self.assertTrue(np.all(utils.logsumexp(logprobs, 1) < 1E-10))

		# test for consistency
		logprobs1 = srbm._ulogprob(examples_vis, examples_hid, all_pairs=True)
		logprobs3 = srbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True) \
		          + srbm._ulogprob_vis(examples_vis).T
		self.assertTrue(np.all(np.abs(logprobs1 - logprobs3) < 1E-10))

		rbm = RBM(srbm.X.shape[0], srbm.Y.shape[0])
		rbm.W = srbm.W
		rbm.b = srbm.b
		rbm.c = srbm.c
		srbm.L *= 0

		logprobs1 = rbm._ulogprob_vis(examples_vis)
		logprobs2 = srbm._ulogprob_vis(examples_vis)
		self.assertTrue(np.all(np.abs(logprobs1 - logprobs2) < 1E-10))

		logprobs1 = rbm._clogprob_hid_vis(examples_vis, examples_hid)
		logprobs2 = srbm._clogprob_hid_vis(examples_vis, examples_hid)
		self.assertTrue(np.all(np.abs(logprobs1 - logprobs2) < 1E-10))



	def test_all_pairs(self):
		srbm = SemiRBM(10, 10)
		srbm.W = np.matrix(np.random.randn(srbm.X.shape[0], srbm.Y.shape[0]))
		srbm.b = np.matrix(np.random.rand(srbm.X.shape[0], 1))
		srbm.c = np.matrix(np.random.randn(srbm.Y.shape[0], 1))

		examples_vis = np.matrix(np.random.rand(srbm.X.shape[0], 100) < 0.5)
		examples_hid = np.matrix(np.random.rand(srbm.Y.shape[0], 100) < 0.5)

		logprob1 = srbm._ulogprob(examples_vis, examples_hid)
		logprob2 = np.diag(srbm._ulogprob(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = srbm._ulogprob(examples_vis[:, 1], examples_hid)
		logprob2 = srbm._ulogprob(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = srbm._ulogprob(examples_vis, examples_hid[:, 1])
		logprob2 = srbm._ulogprob(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)
		
		logprob1 = srbm._clogprob_hid_vis(examples_vis, examples_hid)
		logprob2 = np.diag(srbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True))
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = srbm._clogprob_hid_vis(examples_vis[:, 1], examples_hid)
		logprob2 = srbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True)[1, :]
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)

		logprob1 = srbm._clogprob_hid_vis(examples_vis, examples_hid[:, 1])
		logprob2 = srbm._clogprob_hid_vis(examples_vis, examples_hid, all_pairs=True)[:, 1].T
		self.assertTrue(np.abs(logprob1 - logprob2).sum() < 1E-10)




if __name__ == '__main__':
	unittest.main()
