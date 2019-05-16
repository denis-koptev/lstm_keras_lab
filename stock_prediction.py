import numpy as np
import matplotlib.pyplot as plt
import pandas
import datetime

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

show_plots = True


dateparse = lambda date: pandas.datetime.strptime(date, '%Y-%m-%d')

# Loading dataset
cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
dataset = pandas.read_csv('msft_eod.csv', parse_dates=['Date'], date_parser=dateparse, usecols=cols, index_col='Date')
dataset = dataset.sort_values(by='Date')

print("\nTop 10 lines in dataset:\n")
print(dataset.head(10))


dates = dataset.index
last_date = dates[-1]
print("\n\nLast date: %s" % last_date)
first_date = last_date.replace(year=last_date.year-20)
print("First date: %s" % first_date)

# Take only 20 years
dataset = dataset.loc[first_date:last_date]
dates = dataset.index

# Extract values
# Using 'Open' column
dataset = dataset.iloc[:,0:1].values

if show_plots:
    plt.plot(dates, dataset)
    plt.title("Open values")
    plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

print("\n\nNormalized values:\n")
print(dataset)

if show_plots:
    plt.plot(dates, dataset)
    plt.title("Scaled Open values")
    plt.show()

# Split training and test data
div_idx = int(len(dataset) / 2)
training_set, test_set = dataset[0:div_idx,:], dataset[div_idx:,:]
training_date, test_date = dates[0:div_idx], dates[div_idx:]
print("\n\nLength of training set: %d\nLength of test set: %d" % (len(training_set),len(test_set)))

#print(training_set)

features_set = []
labels = []
features_num = 10
for i in range(features_num, len(training_set)):
    features_set.append(training_set[i-features_num:i,0])
    labels.append(training_set[i,0])

features_set, labels = np.array(features_set), np.array(labels)
print("\n\nSet size: %d" % features_set.shape[0])
print("Features count: %d" % features_set.shape[1])
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

print(len(features_set[0][0]))
print(features_set)

model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))
'''
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
'''
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(features_set, labels, epochs=10, batch_size=32)

test_features=[]
labels=[]
for i in range(features_num, len(test_set)):
    test_features.append(test_set[i-features_num:i,0])
    labels.append(test_set[i])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)

labels = scaler.inverse_transform(labels)

plt.plot(labels, color='blue', label="Actual Open values")
plt.plot(predictions, color='red', label="Predicted Open values")
plt.legend()
plt.show()
'''
plt.plot(labels[:,1], color='blue', label="Actual Open values")
plt.plot(predictions[:,1], color='red', label="Predicted Open values")
plt.legend()
plt.show()
'''
