import sys

sys.path.append('./code')

from deepbelief import DBN, RBM
from numpy import max, matrix, load
from numpy.random import permutation



def main(argv):
	num_visibles = 28 * 28
	num_hiddens = [1000, 1000]
	num_epochs = 50
	batch_size = 100



	# load data samples
	data = load('./data/mnist.npz')['train'] / 255.



	# train 1st layer
	dbn = DBN(RBM(num_visibles, num_hiddens[0]))
	dbn.train(data, num_epochs, batch_size)

	# train 2nd layer
	dbn.add_layer(RBM(num_hiddens[0], num_hiddens[1]))
	dbn.train(data, num_epochs, batch_size, [1E-1, 1E-2])

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
