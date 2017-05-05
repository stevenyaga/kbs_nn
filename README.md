## kbs_nn

### Neural Network to predict the fleet number of KBS Buses

The project is developed using Frappe Python framework using Python 2.7

The following are the steps taken to implement the neural network

1. Pre-process the data. Assume each number plate is linked to a specific year. Add a year column to mimic a time series
1. For each year get the count of buses
1. Take the first 25 records as the training set
1. Take the remaining records as validation set
1. Training patterns to consist of 3 input nodes. 3 for the previous years and the 4th year to act as a target (output node)
1. Train the network and validate using the validation set to get the accuracy
1. To predict for 2017, pass the 2016, 2015 and 2014 as the input values. Use the last derived weights and apply these to the inputs to predict 2017 value

### Code Structure

* [Fleet.csv](data/content/fleet.csv) contains the original raw data before transformation
* [Fleet_Modified.csv](data/content/fleet_modified.csv) contains the data summed up by years. This is the data the system uses for training and validation
* [Predict Fleet](kbs_nn/kbs_nn/doctype/predict_fleet/) directory contains the files that renders on the client. The user will set the neural network values from here
* [neural_networks.py](kbs_nn/neural_network.py) contains the neural network logic
* [utils.py](kbs_nn/utils.py) contains methods to facilitate loading of data from file as well as persisting the results onto files