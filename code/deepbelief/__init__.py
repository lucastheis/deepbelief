"""
Introduction
============
	This package implements algorithms for training and evaluating deep belief
	networks (DBNs). Currently, the following variants of the restricted Boltzmann
	machine are available for constructing DBNs:

		- L{RBM}
		- L{GaussianRBM}
		- L{SemiRBM}

	Also have a look a L{AbstractBM} which specifies much of the interface and
	learning algorithms. In order to evaluate a trained DBN, the L{Estimator}
	class can be used to estimate the the likelihood of the trained model.

Miscellaneous
=============
	For questions, comments or bug reports, contact U{Lucas Theis<mailto:lucas@tuebingen.mpg.de>}.
	This code is published under the U{MIT License<http://www.opensource.org/licenses/mit-license.php>}.

Quick Tutorial
==============
	Assume that C{data} is a 10 by 1000 numpy matrix with real entries containing
	1000 data points.  We want a deep belief network to learn the distribution of
	the data.

		>>> print data.shape
		(10, 1000)

	Import the relevant building blocks
	for constructing deep belief networks.

		>>> from deepbelief import RBM, GaussianRBM, DBN

	Create a first layer with 10 visible units and 50 hidden units. The
	L{GaussianRBM} is suited for modeling continuous data.

		>>> dbn = DBN(GaussianRBM(10, 50))

	Split the data into batches of size 10 and train the first layer for 50
	iterations.

		>>> dbn.train(data, num_epochs=50, batch_size=10)

	Add a second layer to the network.

		>>> dbn.add_layer(RBM(50, 50))

	Train the second layer.

		>>> dbn.train(data, num_epochs=50, batch_size=10)

	Generate another 500 data points by sampling from the trained model.

		>>> samples = dbn.sample(500)
"""
from dbn import DBN
from rbm import RBM
from gaussianrbm import GaussianRBM
from semirbm import SemiRBM
from estimator import Estimator
from mixbm import MixBM
from abstractbm import AbstractBM

__docformat__ = 'epytext'
