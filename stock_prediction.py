#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas
import datetime

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import argparse

# Helper function to parse input arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--rolling', action='store', default=0)
    parser.add_argument('--features', action='store', default=10)
    parser.add_argument('--div_ratio', action='store', default=0.5)
    parser.add_argument('--iterations', action='store', default=100)
    return parser.parse_args()


args = parse_args()
show_plots = args.plots
rolling = int(args.rolling)
features_num = int(args.features)
div_ratio = float(args.div_ratio)
iterations = int(args.iterations)

# Parser for date column
dateparse = lambda date: pandas.datetime.strptime(date, '%Y-%m-%d')

# Loading dataset
cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
dataset = pandas.read_csv('msft_eod.csv', parse_dates=['Date'], date_parser=dateparse, usecols=cols, index_col='Date')
dataset = dataset.sort_values(by='Date')

print("\nTop 10 lines in dataset:\n")
print(dataset.head(10))

# Choose columns for prediction and extract indexes
prediction_columns = ['Open', 'Volume']
cols = [dataset.columns.get_loc(col) for col in prediction_columns]

# Preprocess data using rolling window if specified
if rolling != 0:
    for col in prediction_columns:
        plt.plot(dataset.loc[:, col], color='green')
        dataset.loc[:, col] = dataset.loc[:, col].rolling(window=rolling).mean()
        plt.plot(dataset.loc[:, col], color='red')
        plt.title('%s (original/rolling)' % col)
        plt.show()
    dataset = dataset.iloc[rolling:]

# Take only last 20 years
dates = dataset.index
last_date = dates[-1]
first_date = last_date.replace(year=last_date.year-20)
dataset = dataset.loc[first_date:last_date]
dates = dataset.index

# Extract only columns for prediction
dataset = dataset.iloc[:,cols].values

if show_plots:
    for i in range(len(prediction_columns)):
        plt.plot(dates, dataset[:,i])
        plt.title("%s" % prediction_columns[i])
        plt.show()

# MinMaxScaler to fit values into [0,1] range
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

if show_plots:
    for i in range(len(prediction_columns)):
        plt.plot(dates, dataset[:,i])
        plt.title("%s (scaled)" % prediction_columns[i])
        plt.show()


# Get training and test data

# Split training and test data
div_idx = int(len(dataset) * div_ratio)
training_set, test_set = dataset[0:div_idx,:], dataset[div_idx:,:]
training_date, test_date = dates[0:div_idx], dates[div_idx:]
print("\n\nLength of training set: %d\nLength of test set: %d" % (len(training_set),len(test_set)))

# Fill training data
features_set = []
labels = []
for i in range(features_num, len(training_set)):
    features_set.append(training_set[i-features_num:i,:])
    labels.append(training_set[i,:])

# Transform data in order to pass it to model (3 dimensions)
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], len(cols)))

# Using sequential model (linear stack of layers)
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(features_set.shape[1], len(cols))))

# Adding Dropout (regularization) layer to prevent outfitting
# Dropout will probabilistically remove layer inputs
model.add(Dropout(0.2))

# Add 3 more (hidden) layers
'''
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
'''

# Number of neurons - number of prediction columns
model.add(Dense(units=len(cols)))

# Loss is estimated using mean squared error
# Using 'adam' optimization algorithm instead of regular gradient descent
'''
Instead of adapting the parameter learning rates based on the average first moment (the mean),
Adam also makes use of the average of the second moments of the gradients (the uncentered variance).
'''
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
# epochs - number of iterations on a dataset
# batch_size - he number of samples that will be propagated through the network (for one training)
model.fit(features_set, labels, epochs=iterations, batch_size=32)


# Retrieve and format test sample
test_features=[]
labels=[]
for i in range(features_num, len(test_set)):
    test_features.append(test_set[i-features_num:i,:])
    labels.append(test_set[i,:])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], len(cols)))

# Predict values
predictions = model.predict(test_features)

# Restore values to their original scale (via MinMaxScaler)
predictions = scaler.inverse_transform(predictions)
labels = scaler.inverse_transform(labels)

for i in range(len(cols)):
    col_name = prediction_columns[i]
    plt.plot(labels[:, i], color='blue', label='Actual %s' % col_name)
    plt.plot(predictions[:, i], color='red', label='Predicted %s' % col_name)
    plt.legend()
    plt.show()

