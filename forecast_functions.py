#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np

from pandas_datareader.data import DataReader
from datetime import datetime, timedelta


# In[55]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import backend as K


# In[56]:


import tensorflow as tf
import random as rn
import os
import pickle

# Un intento de hacer resultados reproducibles
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.compat.v1.set_random_seed(1234)


# In[57]:


# Main modules
#import predictions as pr
#import aws_utils as au


# In[58]:


#Folder para guardar los pesos y modelos
#BUCKET_NAME = 'models-in-prod'
FOLDER = 'models'


# In[59]:
def saveScaler(ticker, scaler):
    file_name = ticker + '.pkl'
    with open(os.path.join(FOLDER, file_name), 'wb') as f:
        pickle.dump(scaler, f)
    return

def loadScaler(ticker):
    file_name = ticker + '.pkl'
    with open(os.path.join(FOLDER, file_name), 'rb') as f:
        return pickle.load(f)

def crearSerie(ticker, start='2012-01-01', end=datetime.now(), window_size=60):
    # Obtener cotizaciones desde yahoo finance
    df = DataReader(ticker, data_source='yahoo', start=start, end=end)
    # Vamos a utilizar los valores del cierre
    data = df.filter(['Close'])
    # Obtener valores como array de numpy
    dataset = data.values
    # Obtener el número de filas que se utilizarán para el entrenamiento
    training_data_len = int(np.ceil( len(dataset) * .8 ))
    # Llevar los valores a escala entre 0 y 1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Nos guardamos el scaler para utilizarlo en las predicciones (Agregado V2.0)
    saveScaler(ticker, scaler)
    # Obtener los valores de entrenamiento
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    # Divido los datos de train en x_train e y_train
    # Vamos a usar bloques de train_size
    for i in range(window_size, len(train_data)):
        x_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])
        
    # Se conviertes x_train e y_train a numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Se hace reshape de x_train
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Se crea dataset para testing
    test_data = scaled_data[training_data_len - window_size: , :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(window_size, len(test_data)):
        x_test.append(test_data[i-window_size:i, 0])

    # Convierte x_test a numpy array
    x_test = np.array(x_test)

    # Se hace reshape de x_test
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    
    # Última ventana de tiempo (train_size)
    x_forecast = []
    x_forecast.append(scaled_data[len(dataset) - window_size: , 0])
    x_forecast = np.array(x_forecast)
    x_forecast = np.reshape(x_forecast, (x_forecast.shape[0], x_forecast.shape[1], 1))
   
    return scaler, x_train, y_train, x_test, y_test, data, scaled_data, training_data_len, x_forecast


# In[60]:


def crearModelo(window_size, loss='mae', optimizer='adam', metrics=['mae']):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (window_size, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)    
    return model


# In[61]:


def entrenarModelo(
        tickerPredecir, 
        start='2012-01-01', 
        end=datetime.now(), 
        window_size=60,
        loss='mae', 
        optimizer='adam', 
        metrics=['mae'], 
        epochs=30, 
        batch_size=32,
        validation_split=0.1,
        callbacks=[]):

    print('Crear serie ', tickerPredecir)
    scaler, x_train, y_train, x_test, y_test, data, scaled_data, training_data_len, x_forecast = crearSerie(tickerPredecir, start, end, window_size)
   
    print('Crear modelo ', tickerPredecir)
    modelo = crearModelo(window_size, loss=loss, optimizer=optimizer, metrics=metrics)

    print('Entrenar modelo')
    history = modelo.fit(x_train, y_train,
                          epochs=epochs, batch_size=batch_size,
                          validation_split=validation_split, callbacks=[es])
   
    print('Evaluar modelo multi input')
    modelo.evaluate(x_test, y_test)
    
    print('Testear modelo multi input')
    predictions = modelo.predict(x_test)
    
    # Transformación inversa de los valores predichos a la escala de la serie
    predictions = scaler.inverse_transform(predictions)

    print('')
    print('')
    print('MAE (Mean Absolute Error): ', mean_absolute_error(y_test, predictions))
    print('')
    print('')
    

    # Guardamos los pesos para utilizar en futuras predicciones
    modelo.save_weights(os.path.join(FOLDER, (tickerPredecir + ".h5")))
    #au.upload_to_s3(BUCKET_NAME, FOLDER, file_name) 
    
    return { 
        'modelo': modelo, 
        'scaler': scaler, 
        'x_train': x_train, 
        'y_train': y_train, 
        'x_test': x_test, 
        'y_test': y_test, 
        'data': data, 
        'scaled_data': scaled_data, 
        'training_data_len': training_data_len, 
        'x_forecast': x_forecast }


# In[62]:


def crearModeloMI(models, loss='mae', optimizer='adam', metrics=['mae']):
    model_outputs = []
    model_inputs = []
    
    for model in models.values():
        model_outputs.append(model['modelo'].output)
        model_inputs.append(model['modelo'].input)

    # Concatenamos las salidas
    concat = concatenate(model_outputs)

    # Generamos una única salida con una sola neurona densa lineal
    final_output = Dense(1)(concat)

    # Creamos la instancia del modelo multi-input
    # Al haber múltiples entradas, definimos una lista de inputs
    multi_input_model = Model(model_inputs, final_output)
    
    
    # Compilamos el modelo multi-input
    multi_input_model.compile(loss=loss,
                              optimizer=optimizer,
                              metrics=metrics)    
    return multi_input_model    


# In[63]:


def entrenarModeloMI(
        tickerPredecir, 
        models,
        start='2012-01-01', 
        end=datetime.now(), 
        window_size=60,
        loss='mae', 
        optimizer='adam', 
        metrics=['mae'], 
        epochs=30, 
        batch_size=32,
        validation_split=0.1,
        callbacks=[]):
    model_outputs = []
    model_inputs = []
    arr_train = []
    arr_test = []   
    arr_forecast = []
    
    for model in models.values():
        model_outputs.append(model['modelo'].output)
        model_inputs.append(model['modelo'].input)
        arr_train.append(model['x_train'])
        arr_test.append(model['x_test'])
        arr_forecast.append(model['x_forecast'])

    # Creamos el modelo multi input
    multi_input_model = crearModeloMI(models, loss, optimizer, metrics)
   
    # Datos de la serie a predecir
    y_train = models[tickerPredecir]['y_train']
    y_test = models[tickerPredecir]['y_test']
    
    scaler = loadScaler(tickerPredecir)
    
    data = models[tickerPredecir]['data']
    training_data_len = models[tickerPredecir]['training_data_len']
    
    # Entrenamos el modelo multi-input
    print('Entrenar modelo multi input')
    history = multi_input_model.fit(arr_train, y_train,
                          epochs=epochs, batch_size=batch_size,
                          validation_split=validation_split, callbacks=[es])

    # Guardamos el modelo para futuras predicciones
    #multi_input_model_json = multi_input_model.to_json()
    #with open(os.path.join(FOLDER, (tickerPredecir + "_MI.json")), "w") as json_file:
    #json_file.write(model_json)
    
    # Guardamos los pesos para futuras predicciones
    multi_input_model.save_weights(os.path.join(FOLDER, (tickerPredecir + "_MI.h5")))
    #au.upload_to_s3(BUCKET_NAME, FOLDER, file_name) 
    
    # Evaluamos el modelo en test
    print('Evaluar modelo multi input')
    multi_input_model.evaluate(arr_test, y_test)
    
    # Predecimos
    print('Testear modelo multi input')
    predictions = multi_input_model.predict(arr_test)
    
    # Transformación inversa de los valores predichos a la escala de la serie
    predictions = scaler.inverse_transform(predictions)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    #Obtenemos el MAE
    print('')
    print('')
    print('MAE (Mean Absolute Error): ', mean_absolute_error(y_test, predictions))
    print('')
    print('')
    
    
    
    return multi_input_model, history, valid


# In[64]:


def crearDatosForecast(ticker, window_size=60, loss='mae', optimizer='adam', metrics=['mae']):
    # Preparamos el modelo y los datos utilizados para realizar las predicciones
    # Creamos el modelo
    modelo = crearModelo(window_size, loss, optimizer, metrics)
    # Leemos los pesos guardados anteriormente
    modelo.load_weights(os.path.join(FOLDER,(ticker + '.h5')))
    # Se descargar los datos de la serie pero solo de los últimos dos períodos (largo de la ventana * 2)
    df = DataReader(ticker, data_source='yahoo', start=datetime.now() - timedelta(window_size * 2))
    # Creamos una entrada en un diccionario con la clave del ticker. Incluimos el modelo, la serie con valores del cierre
    # y un vector con los valores del cierre
    data_dic = { 
        'modelo': modelo,
        'data': df.filter(['Close']),
        'dataset': df.filter(['Close']).values}

    # Agregado V2.0
    scaler = loadScaler(ticker)
    data_dic['scaler'] = scaler
    data_dic['scaled_data'] = scaler.fit_transform(data_dic['dataset'])
    return data_dic

def forecast(tickerPredecir, tickersPredictores, forecastDays, window_size=60, loss='mae', optimizer='adam', metrics=['mae']):
    models = {}
    model_outputs = []
    model_inputs = []
   
    # Creamos el modelo, obtenemos los datos de la serie y escalamos los datos 
    models[tickerPredecir] = crearDatosForecast(tickerPredecir, window_size, loss, optimizer, metrics)
   
    for tickerPredictor in tickersPredictores:
        # Creamos el modelo, obtenemos los datos de la serie y escalamos los datos 
        models[tickerPredictor] = crearDatosForecast(tickerPredictor, window_size, loss, optimizer, metrics)
        
    multi_input_model = crearModeloMI(models, 'mae', 'adam', ['mae'])
    multi_input_model.load_weights(os.path.join(FOLDER, (tickerPredecir + '_MI.h5')))
    #au.download_h5py_from_s3(BUCKET_NAME, FOLDER, file_name)
    
    predictions = []

    for i in range(0, forecastDays):
        arr_forecast = []
        
        # Por cada ticker generamos la serie de datos que vamos a utilizar para predecir un día en el futuro
        for key, value in models.items():
            print('ticker: ', key)
            scaled_data = value['scaled_data']
            x_forecast = []
            # Obtenemos los datos previamente escalados para la última ventana de tiempo de la serie
            x_forecast.append(scaled_data[len(scaled_data) - window_size: , 0])
            x_forecast = np.array(x_forecast)
            x_forecast = np.reshape(x_forecast, (x_forecast.shape[0], x_forecast.shape[1], 1))
            arr_forecast.append(x_forecast)
            
            # Si el ticker no es el que vamos a predecir con el modelo multi input, entonces
            # predecimos un día en el futuro para la serie
            # Esto es necesario para seguir prediciendo con el modelo multi input días siguientes ya que el input del modelo multi input
            # debe contener los datos de todas las series utilizadas para la predicción
            if key != tickerPredecir:
                prediction = value['modelo'].predict(x_forecast)
                print(tickerPredictor + ' prediccion (scaled): ', prediction)
                print(tickerPredictor + ' prediccion (inverse): ', value['scaler'].inverse_transform(prediction))
                # Agregamos el valor predecido al final del array de datos escalados. De esta manera cuando volvamos a tomar 
                # la última ventana, estaremos incluyendo el dato predecido
                value['scaled_data'] = np.vstack((value['scaled_data'], prediction))

        # Hacemos los mismo para el modelo multi input
        prediction = multi_input_model.predict(arr_forecast)
        print(tickerPredecir + ' prediccion (scaled): ', prediction)
        print(tickerPredecir + ' prediccion (inverse): ', models[tickerPredecir]['scaler'].inverse_transform(prediction))
        predictions.append(prediction[0])
        models[tickerPredecir]['scaled_data'] = np.vstack((models[tickerPredecir]['scaled_data'], prediction))

    data = models[tickerPredecir]['data']
    date_from = data.index.values[len(data.index.values) - 1:][0] + np.timedelta64(1, 'D')
    days = pd.date_range(date_from, date_from + np.timedelta64(forecastDays - 1, 'D'), freq='D', name='Date')
    dfPredictions = pd.DataFrame(models[tickerPredecir]['scaler'].inverse_transform(predictions), index=days, columns=['Close'])
    
    
    K.clear_session()
    
    return dfPredictions
    


# In[65]:


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)


# In[66]:


window_size = 60
models = {}
