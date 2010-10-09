import numpy as np

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

def binary_numbers(n):
	"""
	Generates an n x 2^n matrix whose columns are filled with binary-encoded
	numbers.

	@type  n: integer
	@param n: number of bits

	@rtype:  matrix
	@return: matrix filled with binary numbers
	"""

	def recursive(B, i=0):
		m = B.shape[1] / 2

		B[i, m:] = 1

		if m > 1:
			recursive(B[:, :m], i + 1)
			recursive(B[:, m:], i + 1)

	B = np.matrix(np.zeros([n, pow(2, n)], 'byte'))
	recursive(B)
	return B



def logsumexp(x, ax=None):
	"""
	Computes the log of the sum of the exp of the entries in x in a numerically
	stable way.

	@type  x: array_like
	@param x: a list or a matrix of numbers

	@type  ax: integer
	@param ax: axis along which the sum is applied

	@rtype:  matrix
	@return: a matrix containing the results
	"""

	x = np.asmatrix(x)
	x_max = x.max(ax)
	return x_max + np.log(np.exp(x - x_max).sum(ax))



def logmeanexp(x, ax=None):
	"""
	Computes the log of the mean of the exp of the entries in x in a numerically
	stable way. Uses logsumexp.

	@type  x: array_like
	@param x: a list or a matrix of numbers

	@type  ax: integer
	@param ax: axis along which the values are averaged

	@rtype:  matrix
	@return: a matrix containing the results
	"""

	x = np.asarray(x)

	if ax is None:
		n = x.size
	else:
		n = x.shape[ax]

	return logsumexp(x, ax) - np.log(n)
