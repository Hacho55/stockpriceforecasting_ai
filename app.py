#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from flask import Flask, request, json, jsonify, render_template, redirect, url_for
from flask_cors import cross_origin
import datetime
import pandas as pd
from pandas_datareader import data
import json
import os


import forecast_functions as fc

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("flask.html")


@app.route('/api/iniciarSimulacion', methods=['POST'])
@cross_origin(headers=['Content-Type']) # Send Access-Control-Allow-Headers
def simulacion():
    datos = request.get_json(force=True)
    
    fechaHasta = datetime.datetime.now().strftime('%Y%m%d')
    fechaDesde = datetime.datetime.now() - pd.to_timedelta(int(datos['diasPrevios']), unit='d')
    df_yahoo = data.DataReader(datos['ticker'], data_source='yahoo', start=fechaDesde.strftime('%Y%m%d'), end=fechaHasta)
    df = pd.DataFrame(df_yahoo[['Close']])
    df.index.rename('date', inplace=True)
    df.rename(columns={'Close' : 'value'}, inplace=True)
    
    df['fecha'] = df.index.strftime('%Y-%m-%d')
    
    
    ticker = datos['ticker']
    days = int(datos['diasForecast'])
    tickerPredictores = (datos['tickerPredictores']).split(',')
    
    print(ticker)
    print(days)
    print(tickerPredictores)
    print(int(datos['diasPrevios']))
    
    forecasting = fc.forecast(ticker, tickerPredictores, days)
    
    #El forecasting me da valores y el indice como fechas
    forecasting['fecha'] = forecasting.index.strftime('%Y-%m-%d')
    forecasting.rename(columns={'Close':'value'}, inplace=True)
    
    df_concat = pd.concat([df,forecasting])
    
    forecasting_dates=forecasting.reset_index(drop=True)
    
    #print(df_concat)
    
    
    
    
    
    return jsonify({'x1' : df_concat['fecha'].to_json(), 'valores': df['value'].to_json(), 'forecast':df_concat['value'].to_json(), 'RMSE': None, 'x2':forecasting_dates['fecha'].to_json(), 'forecast_final':forecasting['value'].to_json(), 'ACF': None, 'PACF': None, 'DIFF': None, 'ADFuller': None })




if __name__ == '__main__':    
    app.run(host='0.0.0.0', debug=True) 
    
    