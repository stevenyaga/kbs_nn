## kbs_nn

kbs_nn

#### License

MIT

h2. Neural Network to predict the fleet number of KBS Buses

The following are the steps taken to implement the neural network

# Pre-process the data. Assume each number plate is linked to a specific year. Add a year column to mimic a time series
# For each year get the count of buses
# Take the first 25 records as the training set
# Take the remaining records as validation set
# Training patterns to consist of 3 input nodes. 3 for the previous years and the 4th year to act as a target (output node)
# Train the network and validate using the validation set to get the accuracy
# To predict for 2017, pass the 2016, 2015 and 2014 as the input values. Use the last derived weights and apply these to the inputs to predict 2017 value

h2. Code Structure

* The raw data is contained in /data/content/fleet.csv
* /data/content/fleet_modified.csv contains the data summed up by years. This is the data the system uses for training and validation
* kbs_nn/kbs_nn/doctype/predict_fleet/ contains the files that renders on the client. The user will set the neural network values from here
* kbs_nn/kbs_nn/neural_networks.py contains the neural network logic
* kbs_nn/kbs_nn/utils.py contains methods to facilitate loading of data from file as well as persisting the results onto files