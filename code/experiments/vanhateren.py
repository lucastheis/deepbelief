import sys

sys.path.append('./code')

from deepbelief import DBN, GaussianRBM, SemiRBM, RBM
from numpy import arange, matrix, dot, log, diag, load
from numpy.linalg import inv



def main(argv):
	# load preprocessed data samples
	data = load('./data/vanhateren.npz')

	# remove DC component (first component)
	data_train = data['train'][1:, :]
	data_test = data['test'][1:, :]



	# create and train 1st layer
	dbn = DBN(GaussianRBM(num_visibles=data_train.shape[0], num_hiddens=100))

	dbn[0].weight_decay = 1E-2
	dbn[0].momentum = 0.9
	dbn[0].sigma = 0.5
	dbn[0].cd_steps = 1

	dbn.train(data_train, num_epochs=50, batch_size=100, learning_rates=[1E-2, 1E-4])



	# evaluate 1st layer
	logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
	loglik = dbn.estimate_log_likelihood(data_test)

	print 'estimated log-partf.:\t', logptf
	print 'estimated log-loss:\t', -loglik



	# create and train 2nd layer
	dbn.add_layer(SemiRBM(num_visibles=100, num_hiddens=100))

	dbn[1].weight_decay = 1E-2
	dbn[1].weight_decay_lateral = 1E-2
	dbn[1].momentum = 0.9
	dbn[1].momentum_lateral = 0.9
	dbn[1].num_lateral_updates = 20
	dbn[1].damping = 0.2
	dbn[1].cd_steps = 1

	dbn.train(data_train, num_epochs=100, batch_size=100, learning_rates=[1E-2, 1E-4])



	# evaluate 2nd layer
	logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
	loglik = dbn.estimate_log_likelihood(data_test, num_samples=50)

	print 'estimated log-partf.:\t', logptf
	print 'estimated log-loss:\t', -loglik

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
