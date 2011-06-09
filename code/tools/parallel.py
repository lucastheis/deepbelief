"""
Tools for simplified parallel processing.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'
__version__ = '0.3.1'

from multiprocessing import Process, Queue, cpu_count
from numpy import iterable, ceil
from numpy.random import rand
from numpy.random import seed as numpy_seed
from random import seed as py_seed
from time import time

def map(function, arguments, max_processes=None):
	"""
	Applies a function to a list of arguments in parallel.

	A single process is created for each argument, except if the argument list
	contains only one element. In this case, no additional process is created.

	Iterable arguments will be assumed to contain multiple arguments to the
	function.

	B{Example:}

	The first example will fork three processes, the second example ten.

		>>> add = lambda x, y: x + y
		>>> print map(add, [(1, 2), (2, 3), (3, 4)])

		>>> square = lambda x: x * x
		>>> print map(square, range(10))

	To restrict the number of processes, use the argument C{max_processes} or
	set C{map.max_processes} to a default value for all calls to L{map}. By
	default, the maximum number of processes is the number of available CPUs.

	@type  function: callable
	@param function: the function or object that will be applied to all arguments

	@type  arguments: list
	@param arguments: a list which contains the arguments

	@type  max_processes: integer
	@param max_processes: restricts the number of processes

	@rtype: list
	@return: an ordered list of return values
	"""

	if not iterable(arguments):
		raise ValueError("Arguments should be stored in a list.")

	if len(arguments) < 1:
		raise ValueError("The argument list should not be empty.")

	if len(arguments) == 1:
		# don't create a process if there is just 1 argument
		if iterable(arguments[0]):
			return [function(*arguments[0])]
		else:
			return [function(arguments[0])]

	if max_processes is None and map.__dict__.has_key('max_processes'):
		# set default number of processes
		max_processes = map.max_processes

	if max_processes is not None and max_processes < len(arguments):
		def wrapper(*arguments):
			"""
			Helper function which processes chunks of arguments at once.
			"""

			results = []
			for args in arguments:
				if not iterable(args):
					args = (args,)
				results.append(function(*args))
			return results

		# apply wrapper to chunks of arguments
		results = map(wrapper, chunkify(arguments, max_processes))

		# flatten list of lists
		return [result for chunk in results for result in chunk]

	def run(function, queue, idx, rnd, *args):
		"""
		A helper function for handling return values. Takes a function and its
		arguments and stores its result in a queue.

		@type  function: function
		@param function: handle to function that will be called

		@type  queue: Queue
		@param queue: stores returned function values

		@type  idx: integer
		@param idx: index used to identify return values

		@type  rnd: float
		@param rnd: a random number to seed random number generator
		"""

		# compute random seed
		rnd_seed = int(1e6 * rnd + 1e6 * time())

		# without it, different processes are likely to use the same seed
		numpy_seed(rnd_seed)
		py_seed(rnd_seed)

		# evaluate function
		queue.put((idx, function(*args)))

	# queue for storing return values
	queue = Queue(len(arguments))

	# generate random numbers to randomize processes
	random_numbers = rand(len(arguments))

	# create processes
	processes = []
	for idx, elem in enumerate(arguments):
		# make sure arguments are packed into tuples
		if not iterable(elem):
			args = (function, queue, idx, random_numbers[idx], elem)
		else:
			args = [function, queue, idx, random_numbers[idx]]
			args.extend(elem)
			args = tuple(args)

		# store and start process
		processes.append(Process(target=run, args=args))
		processes[-1].start()

	# collect results
	results = {}
	for i in range(len(arguments)):
		idx, result = queue.get()
		results[idx] = result

	# wait for processes to finish
	for process in processes:
		process.join()

	return [results[key] for key in sorted(results.keys())]

map.max_processes = cpu_count()



def chunkify(lst, num_chunks):
	"""
	Splits a list into chunks of equal size (or, if that's not possible, into
	chunks whose sizes are as equal as possible). The order of the elements is
	maintained.

	@type  lst: list
	@param lst: a list of arbitrary elements

	@type  num_chunks: integer
	@param num_chunks: number of chunks

	@rtype: list
	@param: a list of lists (the chunks)
	"""

	chunks = []

	N = 0

	for m in range(num_chunks):
		n =  int(ceil(float(len(lst) - m) / num_chunks))
		chunks.append(lst[N:N + n])
		N = N + n

	return chunks

	#return [lst[i::num_chunks] for i in range(num_chunks)]



def chunks(num_indices, num_chunks):
	"""
	Creates chunks of indices for use with L{map}.

	B{Example:}

		>>> def function(indices):
		>>>    for i in indices:
		>>>        do_something(i)
		>>> map(function, chunks(100, 4))

	@type  num_indices: integer
	@param num_indices: number of indices

	@type  num_chunks: integer
	@param num_chunks: number of chunks

	@rtype: list
	@param: a list of lists (the chunks)
	"""

	indices = range(num_indices)

	return [[chunk] for chunk in chunkify(indices, num_chunks)]
