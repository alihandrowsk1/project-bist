import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, GRU, SimpleRNN, Conv1D, Dense, InputLayer
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import sys
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
sys.path.insert(0, 'C:\\Users\\alptu\\PycharmProjects\\pythonProject2\\Bitirme_proje')
import veri_seti as vs
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)



#################################################### VERİ DOSYASINI OKUMA #####################################################
df =vs.df_clean
df.dropna(axis=0, inplace=True)
df.head()


################################################### BAĞIMLI VE BAĞIMSIZ DEĞİŞKEN AYIRMA ################################################

X = df.drop('KCHOL.IS_Close', axis=1)  # Özellikler
y = df['KCHOL.IS_Close']  # Bağımlı değişken

################################################### TRAIN VALIDATION TEST SPLIT ################################################

# Öncelikle veri setini eğitim ve geçici setlere ayırma (%90 eğitim, %10 geçici)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)

# Geçici seti doğrulama ve test setlerine ayırma (%50 doğrulama, %50 test; %10'un %50'si %5 yapar)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

##################################################### ROBUST SCALER ############################################################3

scaler = RobustScaler()

# Eğitim setini fit ve transform etme
X_train_scaled = scaler.fit_transform(X_train)

# Doğrulama ve test setlerini sadece transform etme
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


##################################################### MODEL OLUŞTURMA #############################################################
def build_model(model_type, optimizer):
    model = Sequential()
    model.add(InputLayer((20, 55)))
    if model_type == 'LSTM':
        model.add(LSTM(64, activation='relu'))
    elif model_type == 'GRU':
        model.add(GRU(64, activation='relu'))
    elif model_type == 'CNN':
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
    elif model_type == 'RNN':
        model.add(SimpleRNN(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=[RootMeanSquaredError()])
    return model

##################################################### EARLY STOPPING #############################################################
stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model_type= ['LSTM', 'GRU', 'CNN', 'RNN']
optimizer = ['adam', 'Adadelta']
##################################################### GRID SEARCH ##################################################################
model = KerasRegressor(build_fn= lambda param: build_model(params), verbose=1, validation_data=(X_test_scaled, y_test))

parameters = {'model_type': ['LSTM', 'GRU', 'CNN', 'RNN'],
              'optimizer': ['adam', 'Adadelta'],
              'batch_size': [2, 4, 6, 8, 12, 16, 20],
              'epochs': [4, 8, 12, 50]}

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=2)


grid_search = grid_search.fit(X_train_scaled,y_train,callbacks=[stopper],validation_data=(X_val_scaled,y_val))





############################################### ÇAĞRI ######################################################################


def build_model(model_type, optimizer):
    model = Sequential()
    model.add(InputLayer((20, 7)))
    if model_type == 'LSTM':
        model.add(LSTM(32, activation='relu'))
    elif model_type == 'GRU':
        model.add(GRU(32, activation='relu'))
    elif model_type == 'CNN':
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
    elif model_type == 'RNN':
        model.add(SimpleRNN(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=[RootMeanSquaredError()])
    return model



# EarlyStopping
stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Grid search
model = KerasRegressor(build_fn=build_model, verbose=1, validation_data=(testX, testY))
parameters = {'model_type': ['LSTM', 'GRU', 'CNN', 'RNN'],
              'optimizer': ['adam', 'Adadelta'],
              'batch_size': [2, 4, 6, 8, 12, 16, 20],
              'epochs': [4, 8, 12, 50]}

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=2)


grid_search = grid_search.fit(trainX,trainY,callbacks=[stopper],validation_data=(validationX,validationY))





