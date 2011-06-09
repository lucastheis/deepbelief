import sys

sys.path.append('./code')

from deepbelief import DBN, GaussianRBM, SemiRBM, RBM
from numpy import arange, matrix, dot, log, diag, load, asmatrix, diag
from numpy.linalg import inv



def main(argv):
	# load preprocessed data samples
	print 'loading data...\t',
	data = load('./data/vanhateren.npz')
	print '[DONE]'
	print

	# remove DC component (first component)
	data_train = data['train'][1:, :]
	data_test = data['test'][1:, :]



	# create 1st layer
	dbn = DBN(GaussianRBM(num_visibles=data_train.shape[0], num_hiddens=100))

	# hyperparameters
	dbn[0].learning_rate = 5E-3
	dbn[0].weight_decay = 1E-2
	dbn[0].momentum = 0.9
	dbn[0].sigma = 0.65
	dbn[0].cd_steps = 1
	dbn[0].persistent = True

	# train 1st layer
	print 'training...\t',
	dbn.train(data_train, num_epochs=100, batch_size=100)
	print '[DONE]'

	# evaluate 1st layer
	print 'evaluating...\t',
	logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
	loglik = dbn.estimate_log_likelihood(data_test)
	print '[DONE]'
	print
	print 'estimated log-partf.:\t', logptf
	print 'estimated log-loss:\t', -loglik / data_test.shape[0] / log(2)
	print



	# create 2nd layer
	dbn.add_layer(SemiRBM(num_visibles=100, num_hiddens=100))

	# initialize parameters
	dbn[1].L = dbn[0].W.T * dbn[0].W
	dbn[1].b = dbn[0].W.T * dbn[0].b + dbn[0].c + 0.5 * asmatrix(diag(dbn[1].L)).T
	dbn[1].L = dbn[1].L - asmatrix(diag(diag(dbn[1].L)))

	# hyperparameters
	dbn[1].learning_rate = 5E-3
	dbn[1].learning_rate_lateral = 5E-4
	dbn[1].weight_decay = 5E-3
	dbn[1].weight_decay_lateral = 5E-3
	dbn[1].momentum = 0.9
	dbn[1].momentum_lateral = 0.9
	dbn[1].num_lateral_updates = 20
	dbn[1].damping = 0.2
	dbn[1].cd_steps = 1
	dbn[1].persistent = True

	# train 2nd layer
	print 'training...\t',
	dbn.train(data_train, num_epochs=100, batch_size=100)
	print '[DONE]'

	# evaluate 2nd layer
	print 'evaluating...\t',
	logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
	loglik = dbn.estimate_log_likelihood(data_test, num_samples=100)
	print '[DONE]'
	print
	print 'estimated log-partf.:\t', logptf
	print 'estimated log-loss:\t', -loglik / data_test.shape[0] / log(2)
	print



	# fine-tune with wake-sleep
	dbn[0].learning_rate /= 4.
	dbn[1].learning_rate /= 4.

	print 'fine-tuning...\t',
	dbn.train_wake_sleep(data_train, num_epochs=10, batch_size=10)
	print '[DONE]'

	# reevaluate
	print 'evaluating...\t',
	logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
	loglik = dbn.estimate_log_likelihood(data_test, num_samples=100)
	print '[DONE]'
	print
	print 'estimated log-partf.:\t', logptf
	print 'estimated log-loss:\t', -loglik / data_test.shape[0] / log(2)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
