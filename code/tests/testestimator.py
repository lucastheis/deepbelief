import sys
import unittest
import numpy as np

sys.path.append('./code')

from deepbelief import Estimator, RBM, GaussianRBM, SemiRBM, DBN, utils

class TestEstimator(unittest.TestCase):
	def test_ais_with_rbm(self):
		rbm = RBM(5, 20)
		rbm.W = np.matrix(np.random.randn(5, 20))
		rbm.b = np.matrix(np.random.randn(5, 1))
		rbm.c = np.matrix(np.random.randn(20, 1))

		ais = Estimator(rbm)

		ais_logz = ais.estimate_log_partition_function(100, np.arange(0, 1, 0.001))
		brf_logz = utils.logsumexp(rbm._ulogprob_vis(utils.binary_numbers(rbm.X.shape[0])))

		lower = np.log(np.exp(ais_logz) - 4. * np.sqrt(rbm._ais_var))
		upper = np.log(np.exp(ais_logz) + 4. * np.sqrt(rbm._ais_var))

		self.assertTrue(upper - lower < 1.)
		self.assertTrue(lower < brf_logz and brf_logz < upper)



	def test_ais_with_gaussianrbm(self):
		rbm = GaussianRBM(30, 10)
		rbm.c = np.matrix(np.random.randn(10, 1))
		rbm.W = np.matrix(np.random.randn(30, 10))
		rbm.b = np.matrix(np.random.rand(30, 1))

		ais = Estimator(rbm)

		ais_logz = ais.estimate_log_partition_function(100, np.arange(0., 1., 1E-4))
		brf_logz = utils.logsumexp(rbm._ulogprob_hid(utils.binary_numbers(rbm.Y.shape[0])))

		lower = np.log(np.exp(ais_logz) - 4 * np.sqrt(rbm._ais_var))
		upper = np.log(np.exp(ais_logz) + 4 * np.sqrt(rbm._ais_var))

		self.assertTrue(upper - lower < 1.5)
		self.assertTrue(lower < brf_logz and brf_logz < upper)



	def test_ais_with_semirbm(self):
		rbm = SemiRBM(5, 20)
		rbm.L = np.matrix(np.random.randn(5, 5))
		rbm.L = np.triu(rbm.L) + np.triu(rbm.L).T - 2 * np.diag(np.diag(rbm.L))
		rbm.num_lateral_updates = 5
		rbm.sampling_method = SemiRBM.GIBBS

		ais = Estimator(rbm)

		ais_logz = ais.estimate_log_partition_function(100, np.arange(0, 1, 0.001))
		brf_logz = np.log(np.sum(np.exp(rbm._ulogprob_vis(utils.binary_numbers(rbm.X.shape[0])))))

		lower = np.log(np.exp(ais_logz) - 4 * np.sqrt(rbm._ais_var))
		upper = np.log(np.exp(ais_logz) + 4 * np.sqrt(rbm._ais_var))

		self.assertTrue(upper - lower < 1.)
		self.assertTrue(lower < brf_logz and brf_logz < upper)



	def test_ais_with_semirbm_dbn(self):
		dbn = DBN(RBM(5, 5))
		dbn.add_layer(SemiRBM(5, 5))
		
		ais = Estimator(dbn)
		ais.estimate_log_partition_function(100, np.arange(0, 1, 1E-3), layer=0)
		ais.estimate_log_partition_function(10, np.arange(0, 1, 1E-3), layer=1)

		dbn[0]._brf_logz = utils.logsumexp(dbn[0]._ulogprob_vis(utils.binary_numbers(dbn[0].X.shape[0])))
		dbn[1]._brf_logz = utils.logsumexp(dbn[1]._ulogprob_vis(utils.binary_numbers(dbn[1].X.shape[0])))

		samples = np.concatenate([dbn.sample(25, 100, 20), np.matrix(np.random.rand(5, 25) > 0.5)], 1)

		Y = utils.binary_numbers(dbn[0].Y.shape[0])
		X = utils.binary_numbers(dbn[0].X.shape[0])

		logRy = dbn[1]._ulogprob_vis(Y)
		logQy = utils.logsumexp(dbn[0]._ulogprob(X, Y, all_pairs=True), 0)
		log_sum = utils.logsumexp(dbn[0]._clogprob_hid_vis(samples, Y, all_pairs=True) - logQy + logRy, 1)

		logPx = log_sum + dbn[0]._ulogprob_vis(samples) - dbn[1]._brf_logz
		logPx_ = ais.estimate_log_probability(samples)[0]

		self.assertTrue(np.abs(logPx_.mean() - logPx.mean()) < 0.1)



	def test_ais_with_semirbm_sanity_check(self):
		grbm = GaussianRBM(15, 50)
		grbm.b = np.random.randn(grbm.b.shape[0], 1)
		grbm.c = np.random.randn(grbm.c.shape[0], 1)

		srbm = SemiRBM(50, 20)
		srbm.W = srbm.W * 0.
		srbm.c = srbm.c * 0.
		srbm.L = grbm.W.T * grbm.W
		srbm.b = grbm.W.T * grbm.b + grbm.c + 0.5 * np.matrix(np.diag(srbm.L)).T
		srbm.L = srbm.L - np.matrix(np.diag(np.diag(srbm.L)))

		ais = Estimator(grbm)
		ais.estimate_log_partition_function(num_ais_samples=100, beta_weights=np.arange(0, 1, 1E-3))

		ais = Estimator(srbm)
		ais.estimate_log_partition_function(num_ais_samples=100, beta_weights=np.arange(0, 1, 1E-2))

		glogz = grbm._ais_logz + srbm.Y.shape[0] * np.log(2)
		slogz = srbm._ais_logz + grbm.X.shape[0] * np.log(np.sqrt(2 * np.pi))

		self.assertTrue(np.abs(glogz - slogz) < 1.)



	def test_ais_with_dbn_sanity_check(self):
		dbn = DBN(RBM(5, 20))
		dbn.add_layer(RBM(20, 5))
		dbn.add_layer(RBM(5, 20))

		dbn[0].W = np.matrix(np.random.randn(5, 20))
		dbn[0].b = np.matrix(np.random.rand(5, 1) - 0.5)
		dbn[0].c = np.matrix(np.random.rand(20, 1) - 0.5)

		dbn[1].W = dbn[0].W.T
		dbn[1].b = dbn[0].c
		dbn[1].c = dbn[0].b

		dbn[2].W = dbn[0].W
		dbn[2].b = dbn[0].b
		dbn[2].c = dbn[0].c

		samples = dbn.sample(100)

		ais = Estimator(dbn[0])

		rbm_logz = ais.estimate_log_partition_function(100, np.sqrt(np.arange(0, 1, 0.001)))
		rbm_probs = ais.estimate_log_probability(samples)

		ais = Estimator(dbn)

		dbn_logz = ais.estimate_log_partition_function(100, np.sqrt(np.arange(0, 1, 0.001)))
		dbn_probs = ais.estimate_log_probability(samples)

		self.assertTrue(abs(dbn_logz - rbm_logz) / rbm_logz < 0.02)
		self.assertTrue((np.exp(dbn_probs) - np.exp(rbm_probs)).mean() < 0.02)



	def test_ais_with_dbn(self):
		dbn = DBN(RBM(10, 10))
		dbn.add_layer(RBM(10, 20))

		hidden_states = utils.binary_numbers(dbn[0].Y.shape[0])

		ais = Estimator(dbn)
		ais_logz = ais.estimate_log_partition_function(100, np.sqrt(np.arange(0, 1, 0.001)))
		brf_logz = utils.logsumexp(dbn[1]._ulogprob_vis(hidden_states))

		brf_probs = []
		dbn_probs = []
		dbn_bound = []

		for i in range(50):
			sample = np.matrix(np.random.rand(10, 1)) > 0.5
			
			prob, bound = ais.estimate_log_probability(sample, 200)

			brf_probs.append(utils.logsumexp(dbn[1]._ulogprob_vis(hidden_states) + dbn[0]._clogprob_vis_hid(sample, hidden_states), 1) - brf_logz)
			dbn_probs.append(prob)
			dbn_bound.append(bound)

		self.assertTrue(np.mean(dbn_bound) < np.mean(dbn_probs))
		self.assertTrue(np.abs(np.mean(dbn_probs) - np.mean(brf_probs)) < 0.1)



if __name__ == '__main__':
	unittest.main()
