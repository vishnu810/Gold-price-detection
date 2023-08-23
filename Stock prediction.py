import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime

# load the data
df = pd.read_csv("SBIN")
start_date = "2010-01-01"
end_date = "2023-05-12"
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
df = df.loc[mask]

# create a new dataframe with only the 'Close' column
data = df.filter(['Close'])

# convert the dataframe to a numpy array
dataset = data.values

# get the number of rows to train the model on
training_data_len = np.ceil(len(dataset) * .8)

# scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# create the training data set
training_data_len = int(len(scaled_data) * .8)

train_data = scaled_data[0:training_data_len, :]

# split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
history = model.fit(x_train, y_train, batch_size=16, epochs=100)

# create a new dataframe with only the 'Close' column
new_data = df.filter(['Close'])
# get the last 60 day closing price values and scale the data to be values between 0 and 1
last_60_days = new_data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
# create an empty list
X_test = []
# append the past 60 days
X_test.append(last_60_days_scaled)
# convert the X_test data set to a numpy array
X_test = np.array(X_test)
# reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# get the predicted scaled price
pred_price_scaled = model.predict(X_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price_scaled)

print(f"Predicted price for tomorrow ({datetime.datetime.now().date() + datetime.timedelta(days=1)}): {pred_price[0][0]}")
