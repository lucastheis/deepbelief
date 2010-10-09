import sys
import unittest
import numpy as np

sys.path.append('./code')

from deepbelief import utils

class TestUtils(unittest.TestCase):
	def test_logsumexp(self):
		x = np.pi
		self.assertTrue(np.abs(utils.logsumexp(x) - x) < 1E-10)

		x = -2000.
		self.assertTrue(np.abs(utils.logsumexp(x) - x) < 1E-10)

		x = [-2000., 1.]
		self.assertTrue(np.abs(utils.logsumexp(x) - 1.) < 1E-10)

		x = [[-1000, 100]]
		s = utils.logsumexp(x, 0)
		self.assertTrue(np.abs(s[0, 0] - x[0][0]) < 1E-10)
		self.assertTrue(np.abs(s[0, 1] - x[0][1]) < 1E-10)

		x = [[-1000], [100]]
		s = utils.logsumexp(x, 1)
		self.assertTrue(np.abs(s[0, 0] - x[0][0]) < 1E-10)
		self.assertTrue(np.abs(s[1, 0] - x[1][0]) < 1E-10)



if __name__ == '__main__':
	unittest.main()
