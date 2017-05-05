# -*- coding: utf-8 -*-
# Copyright (c) 2015, Steve Nyaga and contributors
# For license information, please see license.txt

from __future__ import unicode_literals
import frappe
from frappe.model.document import Document
from kbs_nn.neural_network import NeuralNetwork
from frappe.utils import flt, cint

class PredictFleet(Document):
	def do_predict(self):
		cp = NeuralNetwork();
		cp.train({
			'training_size': cint(self.inputs), 				
			'max_iterations': cint(self.iterations),
			'learning_rate':  self.learning_rate,
			'bias': self.bias
		}, do_predict=True)

		return {'accuracy': cp.accuracy, 'predicted_value': cp.predicted_value}		
