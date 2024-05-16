###########################################################
# IMPORT'LAR VE BAZI AYARLAR #
###########################################################

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from veri_seti import df_clean as df

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


###########################################################
# MODELLEME #
###########################################################

# MODEL İÇİN VERİ SETİNİ UYGUN HALE GETİRME #

df.dropna(inplace=True, axis=0)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

training_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, :-1])
    y_train.append(train_data[i, -1])

x_train, y_train = np.array(x_train), np.array(y_train)

features = len(df.columns) - 1
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], features))
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = df["KCHOL.IS_Close"][training_data_len:].values

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :-1])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], features))

model = Sequential()
model.add(SimpleRNN(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

predictions = model.predict(x_test)

scaler.fit(df[["KCHOL.IS_Close"]])
predictions = scaler.inverse_transform(predictions)


###########################################################
# GÖRSELLEŞTİRME #
###########################################################

train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price TRY')
plt.plot(train["KCHOL.IS_Close"])
plt.plot(valid[["KCHOL.IS_Close", 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
