import frappe
from frappe import _
import numpy as np
import utils as utils
import json

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
		self.weights = []	
		self.max_iterations = 10000 #: Maximum iterations to stop endless looping
		self.learning_rate = 0.07
		self.bias = 1
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
		#np.random.seed(1)

		'''set initial weights between 0 and 1. #add one more weight coz of the bias'''
		self.weights = 2 * np.random.random((self.training_size, 1)) - 1	

		utils.write_to_log("Initial weights", single_space=True)
		utils.write_to_log("{0}".format(str(self.weights)), single_space=False)

	
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
		def log_nn_config():
			"""
			Write the neural netwoek config to file to allow user to download the file
			"""
			conf = {}
			conf["inputs"] = self.training_size
			conf["max_iterations"] = self.max_iterations
			conf["bias"] = self.bias
			conf["learning_rate"] = self.learning_rate

			"""write to log"""
			utils.write_to_log('*****NEURAL NETWORK CONFIG*****', single_space=True)
			utils.write_to_log(json.dumps(conf, indent=4))	
		
		if args: #override default settings
			self.override_defaults(args)
		
		utils.clear_log_file()
		log_nn_config()
		''' '''
		utils.write_to_log('*****ERROR VALUE TRACE*****', single_space=False)

		training_set, validation_set = utils.load_data(TRAINING_RECORDS)
		
		'''break the training set into chunks. Add 1 to capture the 4th element as the output'''
		training_chunks = get_chunks(training_set, self.training_size)#[training_set[x:x+training_size + 1] for x in xrange(0, len(training_set), training_size + 1)]
		validation_chunks = get_chunks(validation_set, self.training_size)

		'''initialize weights'''
		self.initialize_weights()				

		inputs, outputs = self.extract_inputs_and_outputs(training_chunks)

		global_error = 0
		
		for x in xrange(self.max_iterations):
			ly0 = inputs #layer 0 is the inputs layer
			
			'''get dot product of the weights and inputs. ly_1 is the hidden layer'''
			ly1 = self.forward_propagate(ly0)
			
			'''get the resulting error'''
			ly1_error = outputs - ly1	
						
			'''gradient descent. learning rate determines how fast we descend'''			
			ly1_change = ly1_error * self.learning_rate * sigmoid(ly1, True)
			
			# get new weights
			new_weights =  np.dot(ly0.T, ly1_change)

			self.update_weights(new_weights)			

			'''log the error results every 10000th iteration'''
			if x > 0 and (x + 1) % 10000 == 0:
				utils.write_to_log("Error after iteration {0}".format(x+1), single_space=True)
				utils.write_to_log("{0}".format(str(ly1_error)), single_space=False)


		#validate model
		self.validate_model(validation_chunks)

		if do_predict:
			'''
			to predict the next fleet of buses, use the most recent self.training_size years which is the same as 
			picking the last chunk in the validation_chunks
			'''		
			inputs = validation_chunks[-1]
			self.predicted_value = self.predict([inputs])

			'''log prediction results'''
			utils.write_to_log("*****PREDICTION*****")
			'''remove the last element which is the target output'''
			utils.write_to_log("Inputs = {0}".format(str(np.delete(inputs, self.training_size))), single_space=True)						
			utils.write_to_log("Trained Weights = {0}".format(str(self.weights)), single_space=True)						
			utils.write_to_log("Predicted value = {0}".format(self.predicted_value), single_space=False)

		self.make_data_available_for_download()

	
	def make_data_available_for_download(self):
		"""
		generates links in the browser to download:
		1. training data by clicking "Download training data" button in the predict fleet page
		2. Log of the activities of the neural network during prediction including the user settings 
		"""	
		'''upload the files...they could have been deleted'''	
		utils.upload_log_file()

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
		return np.ceil(output)[0][0]


	def forward_propagate(self, inputs):
		"""
		gets the weighted sum of all inputs and the NN weights
		"""
		return sigmoid(self.calculate_output(inputs)) # np.dot(inputs, self.weights) + np.random.random())# self.bias)


	def calculate_output(self, inputs, local_error=1):
		"""
		Gets the weighted product of inputs and weights adjusted by the bias
		"""		
		self.bias = (1 * np.random.random((1)) - 1)[0]
		return np.dot(inputs, self.weights) + self.bias * local_error


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
	"""
	whitelists this method so that it can be called using Ajax
	"""
	cp = NeuralNetwork();
	cp.train({
		training_size: 5, 				
		max_iterations: 60000,
		learning_rate: 0.07		
	}, do_predict=True)
