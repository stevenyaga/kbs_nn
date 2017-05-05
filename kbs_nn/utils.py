import frappe
from frappe import _
import numpy as np

def load_data(no_of_training_records):
	"""
	Loads the test / training data
	training_size = number of years to consider when training
	e.g if training_size == 3, then the training set will be x1, x2, x3 and x4
	x4 will be the target output
	"""
	dataset = np.loadtxt(get_content_file_path("fleet_modified.csv"), delimiter=',', skiprows=1)
	#@TODO adjust the TRAINING_RECORDS depending on the training size

	
	training_set = np.array([x[1] for x in dataset][:no_of_training_records]) #get only the count of fleets
	validation_set = np.array([x[1] for x in dataset][no_of_training_records:]) #get only the count of fleets
	return training_set, validation_set
	

def get_processed_file_path(file_name):
	"""
	Returns a file inside the processed content dir
	"""
	path = get_data_dir() + file_name
	return path


def get_content_file_path(file_name):
	"""
	Returns a file inside the raw content dir
	"""
	path = get_data_dir() + 'content/' + file_name
	return path


def get_data_dir(): 
	"""
	return the data dir
	"""   
	import os
	return os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')) + '/'


def get_log_file():
	"""
	returns path to the log file
	"""
	return get_processed_file_path("kbs_nn_log.txt")


def clear_log_file():
	"""
	Empty the log file	
	"""
	log_file = get_log_file()
	with open(log_file, 'w') as df:
		df.write(" ")

def write_to_log(txt, single_space=False):
	"""
	Append to log file
	"""
	log_file = get_log_file()
	with open(log_file, 'a+') as df:
		df.write(txt + ("\r\n\r\n" if not single_space else "\r\n"))

def upload_log_file():
	"""
	Move the log file into a location accessible from the web
	"""
	'''upload the log file'''
	do_upload(get_log_file())

	'''upload the modified .xlsx file'''
	do_upload(get_content_file_path("fleet_modified.xlsx"))

def do_upload(src):
	"""
	Do actual moving into the web accessible location
	"""
	import os, shutil
	dest = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '../../')) + '/sites/'+ frappe.local.site +'/private/files/'
	shutil.copy(src, dest)
