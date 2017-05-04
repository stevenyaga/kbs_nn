import frappe
from frappe import _
import numpy as np
import utils as utils

TRAINING_RECORDS = 25 #number of records to use as training set

class NeuralNetwork:
	def __init__(self):
		"""
		set initial values
		"""
		'''
		trainin_size: #Also can be considered as the number of inputs. see load_data()
		e.g if training_size == 3, then the training set will be x1, x2, x3 and x4
		x4 will be the target output
		'''
		self.training_size = 3 		
		self.hidden_layers = 1 #do not change this
		self.bias = 0.01	
		self.weights = []	
		self.max_iterations = 10000 #: Maximum iterations to stop endless looping
		self.learning_rate = 0.07
		self.accuracy = 0 #updated after the training and validation is completed		
		self.predicted_value = None
	
	
	def override_defaults(self, args):
		"""
		override the default settings
		:param args == args to override
		"""
		self.__dict__.update(args)
		
	
	def initialize_weights(self):
		"""
		set the initial random weights
		"""
		'''seed random numbers for uniformity'''
		np.random.seed(1)

		'''set initial weights between 0 and 1'''
		self.weights = 2 * np.random.random((self.training_size, 1)) - 1	

	
	def update_weights(self, weights):
		"""
		updates the weights to the newly calculated weights
		"""
		self.weights = weights


	def extract_inputs_and_outputs(self, dataset):
		"""
		:param dataset = a dataset in the form of [[x1, x2...x(training_size), output]]
		"""
		"""extract inputs and output components from training chunks"""
		'''extracts self.training_size elements'''
		inputs = np.array([x[0:self.training_size] for x in dataset])

		'''extracts the last element since its the output'''
		outputs = np.array([x[self.training_size:] for x in dataset])

		return inputs, outputs

	
	def train(self, args, do_predict=False):
		"""
		Train the network 
		"""
		if args: #override default settings
			self.override_defaults(args)

		training_set, validation_set = utils.load_data(TRAINING_RECORDS)
		
		'''break the training set into chunks. Add 1 to capture the 4th element as the output'''
		training_chunks = get_chunks(training_set, self.training_size)#[training_set[x:x+training_size + 1] for x in xrange(0, len(training_set), training_size + 1)]
		validation_chunks = get_chunks(validation_set, self.training_size)

		'''initialize weights'''
		self.initialize_weights()		

		inputs, outputs = self.extract_inputs_and_outputs(training_chunks)
		
		for x in xrange(self.max_iterations):
			ly0 = inputs #layer 0 is the inputs layer
			
			#print "ly0 = {0}".format(ly0)
			#print "weights = {0}".format(self.weights)
			'''get dot product of the weights and inputs. ly_1 is the hidden layer'''
			ly1 = self.forward_propagate(ly0)

			'''get the resulting error'''
			ly1_error = outputs - ly1		

			'''gradient descent. learning rate determines how fast we descend'''			
			ly1_change = ly1_error * self.learning_rate * sigmoid(ly1, True)

			#print "ly1_change = {0}".format(ly1_change)

			#import pdb; pdb.set_trace()
			# get new update weights
			new_weights =  np.dot(ly0.T, ly1_change)

			self.update_weights(new_weights)

		#validate model
		self.validate_model(validation_chunks)

		if do_predict:
			'''
			to predict the next fleet of buses, use the most recent self.training_size years which is the same as 
			picking the last chunk in the validation_chunks
			'''		
			self.predicted_value = self.predict([validation_chunks[-1]])
			print "Ouput after prediction"
			print self.predicted_value
	

	def validate_model(self, validation_set):
		"""
		checks the accuracy with which it can predict output given inputs
		"""
		inputs, targets = self.extract_inputs_and_outputs(validation_set)
		'''forward propagate the inputs to get inputs using the saved weights'''
		outputs = self.forward_propagate(inputs)
		'''get the resulting error'''
		error = targets - outputs
		self.accuracy = np.mean(error)


	def predict(self, dataset):
		"""
		predict a value based on the input and the weights as saved after training the model		
		"""
		inputs, targets = self.extract_inputs_and_outputs(dataset)
		output = self.forward_propagate(inputs)		
		return output[0][0]


	def forward_propagate(self, inputs):
		"""
		gets the weighted sum of all inputs and the NN weights
		"""
		return sigmoid(np.dot(inputs, self.weights))


	# def get_training_and_validation_chunks(dataset):
	# 	"""
	# 	breaks down the dataset into training and validation chunks consisting of inputs and a target
	# 	"""
	# 	training_chunks, validation_chunks = []
	# 	for x in xrange(len(dataset)):			
	# 		if x % training_size == 0:
	# 			'''pick training_size + 1 elements'''
	# 			if x <= TRAINING_RECORDS:
	# 				training_chunks.append(dataset[x-self.training_size:(x / self.training_size)* self.training_size + 1])	
	# 			else:
	# 				validation_chunks.append(dataset[x-self.training_size:(x / self.training_size)* self.training_size + 1])	

	# 	'''remove those that might have less length than training_size + 1'''
	# 	training_chunks = [x for x in training_chunks if len(x) == self.training_size + 1]
	# 	validation_chunks = [x for x in validation_chunks if len(x) == self.training_size + 1]
	# 	return training_chunks, validation_chunks


def get_chunks(dataset, no_of_inputs):
	"""
	breaks down the dataset into chunks consisting of inputs and a target
	"""
	chunks = []
	for x in xrange(len(dataset)):
		if x % no_of_inputs == 0:
			'''pick no_of_inputs + 1 elements'''
			chunks.append(dataset[x- no_of_inputs:(x / no_of_inputs)* no_of_inputs + 1])	

	'''remove those that might have less length than training_size + 1'''
	return [x for x in chunks if len(x) == no_of_inputs + 1]


def sigmoid(x, get_derivative=False):
	"""
	sigmoid function to ensure continuous values for activation
	"""
	if(get_derivative == True):
		return x*(1-x)
	return 1 / (1+np.exp(-x))


@frappe.whitelist()
def predict_kbs_fleet():
	cp = NeuralNetwork();
	cp.train({
		training_size: 5, 				
		bias: 0.03,			
		max_iterations: 60000,
		learning_rate: 0.07		
	}, do_predict=True)

# def do_predict(iterations=2, hidden_layers=1, training_size=3):
# 	"""
# 	Do prediction of the following year fleet
# 	:param iterations: Maximum iterations to stop endless looping
# 	:param hidden_layers: Number of hidden layers. 
# 	:param training_size: Also can be considered as the number of inputs. see load_data()
# 	:return:
# 	"""

# 	'''seed random numbers for uniformity'''
# 	np.random.seed(1)	

# 	training_set, validation_set = load_data()
	
# 	'''break the training set into chunks. Add 1 to capture the 4th element as the output'''
# 	training_chunks = get_training_chunks(training_set, training_size)#[training_set[x:x+training_size + 1] for x in xrange(0, len(training_set), training_size + 1)]
	
# 	'''set initial weights between 0 and 1'''
# 	layer_weights = 2 * np.random.random((training_size, 1)) - 1	

# 	X = np.array([x[0:training_size] for x in training_chunks])
# 	Y = np.array([x[training_size:] for x in training_chunks])

# 	for x in xrange(iterations):
# 		'''@TODO train with all training chunks'''
# 		#output = [x[training_size] for x in training_chunks]
# 		#for chunk in training_chunks:

# 		'''inputs are the first 3 elements. the 4 element is the target'''
# 		layer_0 = X #np.array(training_chunks[0][:3])

# 		'''get dot product of the weights and number of fleets'''
# 		layer_1 = sigmoid(np.dot(layer_0, layer_weights))
		
# 		layer_1_error = Y - layer_1

# 		#gradient descent
# 		layer1_change = layer_1_error * sigmoid(layer_1, True)

# 		#modify weights
# 		layer_weights += np.dot(layer_0.T, layer1_change)

# 	print "Output after training"
# 	print layer_1

# def sigmoid(x, derivative=False):
# 	"""
# 	sigmoid function to ensure continuous values
# 	"""
# 	if(derivative == True):
# 		return x*(1-x)
# 	return 1 / (1+np.exp(-x))


# def validate_code():
# 	import numpy as np

# 	# sigmoid function
# 	def nonlin(x,deriv=False):
# 		if(deriv==True):
# 		    return x*(1-x)
# 		return 1/(1+np.exp(-x))

# 	# input dataset
# 	X = np.array([  [0,0,1],
# 	            [0,1,1],
# 	            [1,0,1],
# 	            [1,1,1] ])

# 	# output dataset            
# 	y = np.array([[0,0,1,1]]).T

# 	# seed random numbers to make calculation
# 	# deterministic (just a good practice)
# 	np.random.seed(1)

# 	# initialize weights randomly with mean 0
# 	syn0 = 2*np.random.random((3,1)) - 1
# 	for iter in xrange(10000):

# 		# forward propagation
# 		l0 = X
# 		l1 = nonlin(np.dot(l0,syn0))

# 		# how much did we miss?
# 		l1_error = y - l1
# 		# multiply how much we missed by the 
# 		# slope of the sigmoid at the values in l1
# 		l1_delta = l1_error * nonlin(l1,True)

# 		# update weights
# 		syn0 += np.dot(l0.T,l1_delta)

# 	print "Output After Training:"
# 	print l1


# def validate_code_mine():
# 	import numpy as np

# 	# sigmoid function
# 	def nonlin(x,deriv=False):		
# 		if(deriv==True):
# 		    return x*(1-x)
# 		return 1/(1+np.exp(-x))

# 	# input dataset
# 	X = np.array([  
# 				[0,0,100],
# 	            [0,12,13],
# 	            [41,0,18],
# 	            [51,16,17] ])

# 	# output dataset            
# 	y = np.array([[0,0,1,1]]).T

# 	# seed random numbers to make calculation
# 	# deterministic (just a good practice)
# 	np.random.seed(1)

# 	# initialize weights randomly with mean 0
# 	syn0 = 2*np.random.random((3,1)) - 1
# 	import pdb; pdb.set_trace()
# 	for iter in xrange(10000):

# 		# forward propagation
# 		l0 = X
# 		l1 = nonlin(np.dot(l0,syn0))

# 		# how much did we miss?
# 		l1_error = y - l1

# 		# multiply how much we missed by the 
# 		# slope of the sigmoid at the values in l1
# 		l1_delta = l1_error * nonlin(l1,True)

# 		# update weights
# 		syn0 += np.dot(l0.T,l1_delta)

# 	print "Output After Training:"
# 	print l1
