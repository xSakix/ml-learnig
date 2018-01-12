from pandas_datareader import data as data_reader
import pandas
import matplotlib.pyplot as plt
import os
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import losses

file = 'open_data.csv'

if os.path.isfile(file):
    data = pandas.read_csv(file)
else:
    data = data_reader.DataReader(['BTC-USD'], 'yahoo', '2017-01-01', '2018-01-11')
    data.ix['Open'].to_csv(file)
    data = pandas.read_csv(file)

if data['Date'][0] > data['Date'][len(data['Date']) - 1]:
    rows = []
    rows2 = []
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

# '1' - up, '0' - down
labels = data.pct_change()
labels = labels.iloc[2:]
labels[labels > 0.] = 1
labels[labels < 0.] = 0
labels = np.array(labels['BTC-USD'])


#prepare data
# price, avg price, rank,  mean,median, std, quantile(0.9),skew, kurt
train_data = []
#rol = data.cumsum().rolling(window=30).sum()
rank = data.rank()

for i in data.index:
    if i == 0 or i == 1:
        continue
    price_before = data.iloc[i-1]['BTC-USD']
    price = data.iloc[i]['BTC-USD']
    day_data = []
    day_data.append(price_before)
    day_data.append(price)
    day_data.append(price/data.iloc[:i]['BTC-USD'].sum())
    day_data.append(rank.iloc[i]['BTC-USD'])
    #day_data.append(rol.iloc[i]['BTC-USD'])
    day_data.append(data.iloc[:i].mean()['BTC-USD'])
    day_data.append(data.iloc[:i].median()['BTC-USD'])
    day_data.append(data.iloc[:i].std()['BTC-USD'])
    day_data.append(data.iloc[:i].quantile(0.9)['BTC-USD'])
    day_data.append(data.iloc[:i].skew()['BTC-USD'])
    day_data.append(data.iloc[:i].kurt()['BTC-USD'])
    #print(day_data)
    day = np.array(day_data)
    day[np.isnan(day)] = 0.
    train_data.append(day)

# zip with labels into one list
print(labels)
print(train_data)

# print(len(train_data))
data_set = train_data
# labels = labels[1:]
print(len(data_set))


train_data = np.array(data_set[:int(len(data_set)/3)])
val_data = np.array(data_set[int(len(data_set)/3):2*int(len(data_set)/3)])
test_data = np.array(data_set[2*int(len(data_set)/3):])

train_labels = np.array(labels[:int(len(labels)/3)])
val_labels = np.array(labels[int(len(labels)/3):2*int(len(labels)/3)])
test_labels = np.array(labels[2*int(len(labels)/3):])

print(np.shape(train_data))

model = Sequential()
model.add(Dense(21, activation='sigmoid',input_dim = 10))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss=losses.mse,
              optimizer=SGD(),
              metrics=['accuracy'])

history = model.fit(train_data,train_labels,
                    epochs=1000,
                    verbose=1,
                    validation_data=(val_data, val_labels))
score = model.evaluate(test_data,test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# plt.plot(labels, 'o')
# plt.plot(train_data)
# plt.show()
