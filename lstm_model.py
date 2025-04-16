import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib; matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings('ignore')

# Load the data from the csv file
data = pd.read_csv('GOOG.csv')

data.isnull().sum()

def data_understanding(dataframe, num=5, plot=False):
    print('########## Dataset Shape ##########')
    print('Number of rows:', dataframe.shape[0], '\nNumber of columns:', dataframe.shape[1])
    print('\n')
    print(f'########## First {num} Rows ##########')
    print(dataframe.head(num))
    print('\n')
    print('########## Column Names ##########')
    print(dataframe.columns)
    print('\n')
    print('########## Unique & Null values ##########')
    data = pd.DataFrame(index=dataframe.columns)
    data["Unique_values"] = dataframe.nunique()
    data["Null_values"] = dataframe.isnull().sum()
    print(data)
    print('\n')
    print('########## Variable Types ##########')
    print(dataframe.info())
    print('\n')
    print('########## Summary Statistics ##########')
    print(dataframe.describe().transpose())
    if plot:
        for col in dataframe.columns:
            if dataframe[col].dtype != "O":
                sns.histplot(x=col, data=dataframe, kde=True, bins=100)
                plt.show(block=True)
    if plot:
        for col in dataframe.columns:
            if dataframe[col].nunique() < 60:
                sns.countplot(x=col, data=dataframe)
                plt.xticks(rotation=90)
                plt.show(block=True)

data_understanding(data,5)

# Data Preprocessing
# Symbol column does not provide any useful information, so we can drop it
data = data.drop(['symbol'], axis=1)

# Date column is not in the right format, so we need to convert it to datetime
data["date"] = pd.to_datetime(data["date"], utc=True)
#data["date"] = data["date"].dt.strftime('%Y-%m-%d').to_datetime()

data.info()
data.isnull().sum()
data["date"][1]
# '2016-06-15'
# We can see that the date is in the format 'YYYY-MM-DD'

# Preliminary analysis of the data to understand the trend
plt.plot(data["date"], data["close"])
plt.title("Google Stock Price")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.show()

# We can see that the stock price has been increasing over the years

# First, make a copy of our data and be sure that the data is sorted by date
time_data = data.copy()
time_data = time_data.sort_values("date")

# Next, we need to set the date as the index of the dataframe
time_data = time_data.set_index("date")
time_data.isnull().sum()

time_data.head()

def plot_time_series(df, feature_cols):
    fig, axes = plt.subplots(len(feature_cols), 1, figsize=(10, 8), sharex=True)

    for ax, feature in zip(axes, feature_cols):
        ax.plot(df[feature], label=feature, linewidth=2)
        ax.set_ylabel(feature)
        ax.legend()

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()

plot_time_series(time_data, time_data.columns)

# For the analysis, we will use the close, high, low, open, and volume columns
time_data = time_data[["close", "high", "low", "open", "volume"]]
time_data.head()

# We will resample the data to weekly data

time_data_close = time_data["close"].resample("W").mean()
time_data_close[:7]
time_data_close.dropna(inplace=True)

# Stationary test (Dickey-Fuller)
def is_stationary(data_series):
    # "HO: Non-stationary"
    # "H1: Stationary"
    p_value = sm.tsa.stattools.adfuller(data_series)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(time_data_close)

def ts_decompose(data_series, model="additive", stationary=False, save=False):
    result = seasonal_decompose(data_series, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(data_series, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if save:
        plt.savefig("Stl_decomposition.png")


    if stationary:
        is_stationary(data_series)

ts_decompose(time_data_close, save=True)

# train-test split
# We will split the data into training and testing data
# We will use the first 80% of the data for training and the remaining 20% for testing

len(time_data)

training_size = int(len(time_data) * 0.8)
testing_size = len(time_data) - training_size

training_data = time_data.iloc[0:training_size]
testing_data = time_data.iloc[training_size:]

training_set = training_data[["close"]].values
test_set = testing_data[["close"]].values


scaler = MinMaxScaler(feature_range=(0,1))
training_set_scaled = scaler.fit_transform(training_set)

timesteps = 60

X_train = []
y_train = []
for i in range(timesteps, len(training_set)):
    X_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

data_total = time_data['close'].values
#data_total.shape
#data_total[:5]


inputs = data_total[len(data_total)-len(test_set) - 60:]
#inputs[:5]

inputs = inputs.reshape(-1,1)
#inputs[:5]
inputs  = scaler.transform(inputs)


X_test = []
for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# LSTM Model
# We will create the LSTM model

#
# model = Sequential()
# model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=100, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=100, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(units = 25))
# model.add(Dense(units=1))
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.summary()

# The LSTM architecture
Model = Sequential()
# First LSTM layer with Dropout regularisation
Model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1],1)))
Model.add(Dropout(0.2))
# Second LSTM layer
Model.add(LSTM(units = 100, return_sequences = True))
Model.add(Dropout(0.2))
# Third LSTM layer
Model.add(LSTM(units = 100, return_sequences = True))
Model.add(Dropout(0.2))
# Fourth LSTM layer
##add 4th lstm layer
#Model.add(layers.LSTM(units = 100))
#Model.add(layers.Dropout(rate = 0.2))

Model.add(layers.LSTM(units = 100, return_sequences = False))
Model.add(layers.Dropout(rate = 0.2))
Model.add(layers.Dense(units = 25))
Model.add(layers.Dense(units = 1))
# The output layer
Model.add(Dense(units = 1))

Model.summary()

Model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# We will train the LSTM model

epochs = 20
batch_size = 32

history = Model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Training loss
# We will visualize the training loss
loss = history.history['loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training loss', size=15, weight='bold')
plt.legend(loc=0)
plt.show()

# Predictions

predicted_stock_price = Model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# We will visualize the results of the LSTM model and compare the real stock price with the predicted stock price

plt.figure(figsize=(10,6))
plt.plot(test_set, color='r',label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='b',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.grid(True)
plt.legend()
plt.show()

plt.savefig('stock_price_prediction.png')

