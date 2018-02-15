from pandas_datareader import data as data_reader
import pandas
import matplotlib.pyplot as plt
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import losses


def relu_net(input_dim=10):
    model = Sequential()
    L1 = layer_size(input_dim)
    model.add(Dense(L1, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    L2 = layer_size(L1)
    model.add(Dense(L2, activation='relu'))
    model.add(Dropout(0.5))
    L3 = layer_size(L2)
    model.add(Dense(L3, activation='relu'))
    model.add(Dropout(0.5))
    L4 = layer_size(L3)
    model.add(Dense(L4, activation='relu'))
    model.add(Dropout(0.5))
    L5 = layer_size(L4)
    model.add(Dense(L5, activation='relu'))
    model.add(Dropout(0.5))
    L6 = layer_size(L5)
    model.add(Dense(L6, activation='relu'))
    model.add(Dropout(0.5))
    L7 = layer_size(L6)
    model.add(Dense(L7, activation='relu'))
    model.add(Dropout(0.5))
    L8 = layer_size(L7)
    model.add(Dense(L8, activation='relu'))
    model.add(Dropout(0.5))
    L9 = layer_size(L8)
    model.add(Dense(L9, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def layer_size(input_dim):
    return input_dim * 2 + 1


def sigmoid_net(input_dim=10):
    model = Sequential()
    model.add(Dense(layer_size(input_dim), activation='sigmoid', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def tanh_net(input_dim=10):
    model = Sequential()
    L1 = layer_size(input_dim)
    model.add(Dense(L1, activation='tanh', input_dim=input_dim))
    model.add(Dropout(0.5))
    # L2 = layer_size(L1)
    # model.add(Dense(L2, activation='tanh', input_dim=input_dim))
    # model.add(Dropout(0.5))
    # L3 = layer_size(L2)
    # model.add(Dense(L3, activation='tanh', input_dim=input_dim))
    # model.add(Dropout(0.5))
    # L4 = layer_size(L3)
    # model.add(Dense(L4, activation='tanh', input_dim=input_dim))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    return model


def load_data():
    file = 'open_data.csv'
    if os.path.isfile(file):
        data = pandas.read_csv(file)
    else:
        data = data_reader.DataReader(['BTC-USD'], 'yahoo', '2010-01-01', '2018-01-11')
        data.ix['Open'].to_csv(file)
        data = pandas.read_csv(file)
    if data['Date'][0] > data['Date'][len(data['Date']) - 1]:
        rows = []
        for i in reversed(data.index):
            row = [data[key][i] for key in data.keys()]
            rows.append(row)

        data = pandas.DataFrame(rows, columns=data.keys())
    del data['Date']
    indexes = []
    for key in data.keys():
        for i in data[key].index:
            val = data[key][i]
            try:
                if np.isnan(val) and not indexes.__contains__(i):
                    indexes.append(i)
            except TypeError:
                if not indexes.__contains__(i):
                    indexes.append(i)
    data.drop(indexes, inplace=True)
    return data


def create_labels():
    # '1' - up, '0' - down
    labels = data.pct_change()
    labels = labels.iloc[2:]
    labels[labels['BTC-USD'] > 0.1] = 1
    labels[labels['BTC-USD'] <= 0.1] = 0
    labels = np.array(labels['BTC-USD'])
    return labels


def prepare_data_set(data):
    # prepare data
    # source : https://www.safaribooksonline.com/library/view/python-for-finance/9781491945360/ch01.html
    log_returns = np.log(data['BTC-USD'] / data['BTC-USD'].shift(1))
    # log_returns = data['BTC-USD']
    sma = pandas.rolling_mean(log_returns, window=30) * np.sqrt(30)
    median = pandas.rolling_median(log_returns, window=30) * np.sqrt(30)
    std = pandas.rolling_std(log_returns, window=30) * np.sqrt(30)

    # source https://www.quantinsti.com/blog/build-technical-indicators-in-python/
    cci = (log_returns - pandas.rolling_mean(log_returns, 30) * np.sqrt(30)) / (
            0.015 * pandas.rolling_std(log_returns, 30) * np.sqrt(30))
    ewma = pandas.ewma(log_returns, span=30) * np.sqrt(30)
    roc = log_returns.diff(30) / log_returns.shift(30)
    upperbb = sma + (2 * std)
    lowerbb = sma - (2 * std)

    f, p = plt.subplots(3, 3)
    p[0, 0].plot(log_returns)
    p[0, 0].set_title('log returns')
    # p[0, 0].set_title('btc data')
    p[0, 1].plot(sma)
    p[0, 1].set_title('sma')
    p[0, 2].plot(median)
    p[0, 2].set_title('median')
    p[1, 0].plot(std)
    p[1, 0].set_title('std')
    p[1, 1].plot(cci)
    p[1, 1].set_title('cci')
    p[1, 2].plot(ewma)
    p[1, 2].set_title('ewma')
    p[2, 0].plot(roc)
    p[2, 0].set_title('roc')
    p[2, 1].plot(upperbb)
    p[2, 1].plot(lowerbb)
    p[2, 1].set_title('upperbb and lowerbb')
    plt.show()

    train_data = []
    for i in data.index:
        if i == 0:
            continue
        day_data = [
            log_returns[i],
            sma[i],
            median[i],
            std[i],
            cci[i]
        ]
        day = np.array(day_data)
        day[np.isnan(day)] = 0.
        train_data.append(day)
    return np.array(train_data)


def load_data_from_data_set(data_set, labels, size):
    indexes = np.random.choice(range(len(data_set)), size=size)

    return data_set[indexes], labels[indexes]


data = load_data()
labels = create_labels()
train_data = prepare_data_set(data)
data_set = train_data[:-1]
train_data, train_labels = load_data_from_data_set(data_set, labels, int(2 * len(data) / 3))
val_data, val_labels = load_data_from_data_set(data_set, labels, int(len(data) / 3))
test_data, test_labels = load_data_from_data_set(data_set, labels, int(len(data) / 3))

# model = relu_net(len(train_data[0]))
model = sigmoid_net(len(train_data[0]))
# model = tanh_net(len(train_data[0]))

model.summary()

model.compile(loss=losses.binary_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=1000,
                    verbose=1,
                    validation_data=(val_data, val_labels))

# print(history.history.keys())
# 'loss', 'val_acc', 'acc', 'val_loss'
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.legend(['acc', 'loss'])
plt.show()

score, acc = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score)
print('Test accuracy:', acc)
