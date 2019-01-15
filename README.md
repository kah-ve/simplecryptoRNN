simplecryptoRNN 

modified some things. 

mostly from: https://www.youtube.com/watch?v=ne-dpRdNReI


```python

import pandas as pd
import os
from sklearn import preprocessing  # will throw warning about imp on cloudpickle
import numpy as np
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
# To Use CuDNNLSTM: modify the LSTMs below, delete activation functions as well
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# CONSTANTS
SEQ_LEN = 60  # last 60 minutes of pricing data
FUTURE_PERIOD_PREDICT = 3  # want to predict out 3 minutes
RATIO_TO_PREDICT = 'LTC-USD'  # which ratio we want to predict on
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):
    if float(future) > float(current):
        return 1  # should buy
    else:
        return 0


def scaling(df):
    df = df.drop('future', 1)  # don't need this column anymore, don't train on this

    # scale columns by percent change
    for col in df.columns:
        # don't scale target column
        if col != 'target':

            df[col] = df[col].pct_change()  # normalizing by the pct change
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)  # scaling between 0 and 1
    df.dropna(inplace=True)  # catch any na caused by scaling

    return df


# def sliding_window(dataset, seq_len, stride):
#     n_samples, d = dataset.shape
#     data_slide = np.zeros((int((n_samples - seq_len) / stride) + 1, seq_len, d))
#     k = 0

#     for i in range(0, n_samples - seq_len, stride):
#         data_slide[k, :, :] = dataset[i:i + seq_len, :]
#         k = k + 1


def preprocess_df(df):

    # the sliding window for sequential data
    prev_days = deque(maxlen=SEQ_LEN)

    # where the sliding windows are appended
    sequential_data = []

    # .values returns a numpy representation of the DataFrame
    for i in df.values:
        prev_days.append(i[:-1])  # leave target column out of prev_day data

        # prev_days is now a window as large as we want it to be
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])  # i[-1] is the target column label = y

    random.shuffle(sequential_data)

    # balance the data
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    # what is minimum length of two
    lower = min(len(buys), len(sells))

    # rebalanced to minimum length
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells

    # shuffle buys and sells from being ordered
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


# crypto_data should be in same folder
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


# read in the data
main_df = pd.DataFrame()
ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={'close': f'{ratio}_close', 'volume': f'{ratio}_volume'}, inplace=True)

    df.set_index('time', inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# create future column that is the price of LTC-USD shifted back 3 minutes
main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)

# use classifier to make a target column that has value 1 when value is larger in future
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
# print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head(10))

# use last 5% of data for testing
times = sorted(main_df.index.values)
last_5pct = int(-len(times) * 0.05)  # index before last 5% of data is test
last_10pct = int(-len(times) * 0.1)  # index from 10% of to 5% of data is validation

# the testing data
testing_df = main_df[last_5pct:]
testing_df = scaling(testing_df)

# keep validation and training data together when first scaling
main_df = main_df[:last_5pct]
main_df = scaling(main_df)

# split into train and validation
training_df = main_df[:last_10pct]
validation_df = main_df[last_10pct:last_5pct]

# preprocess the data
train_x, train_y = preprocess_df(training_df)
validation_x, validation_y = preprocess_df(validation_df)

# some statistics
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"don't buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION don't buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation='tanh', input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation='tanh', input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr = 0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

# Save the Models
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))  # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint])

# check tensorboard with: tensorboard --logdir=logs 
# need to be in directory where logs would be


```
