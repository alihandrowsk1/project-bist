###########################################################
# IMPORT'LAR VE BAZI AYARLAR #
###########################################################

import warnings
import pandas as pd
from datetime import datetime
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from hyperopt import hp, fmin, tpe, Trials
import veri_seti as vs

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


###########################################################
# MODELLEME #
###########################################################


def dataf_for_prophet(dataframe, y):
    dataframe["ds"] = dataframe.index
    dataframe["y"] = dataframe[y]
    dataframe.drop([y], axis=1, inplace=True)
    dataframe.columns = dataframe.columns.map('_'.join)
    dataframe.rename(columns={"y_": "y", "ds_": "ds"}, inplace=True)
    return dataframe


df = dataf_for_prophet(vs.df_clean, ("KCHOL.IS", "Close"))


# EN İYİ ÇAPRAZ DOĞRULAMA DEĞERLERİNİ BULMA #

def objective0(params):
    initial = params['initial']
    period = params['period']
    horizon = params['horizon']

    model0 = Prophet()
    model0.fit(df)
    df_cv0 = cross_validation(model0, initial=initial, period=period, horizon=horizon)
    df_p0 = performance_metrics(df_cv0)
    return -df_p0['rmse'].values[0]  # Çapraz doğrulama performansını maksimize et


hyper_iph = {
    'initial': hp.choice('initial', ["45 days", '90 days', '180 days',
                                     '365 days', '730 days', '1095 days']),
    'period': hp.choice('period', ['45 days', '90 days', '180 days',
                                   '365 days', '730 days', '1095 days']),
    'horizon': hp.choice('horizon', ['45 days', '90 days',
                                     '180 days', '365 days', '730 days'])
}

trials0 = Trials()
last_iph = fmin(objective0, hyper_iph, algo=tpe.suggest, max_evals=10, trials=trials0)

best_iph = {
    'initial': ['45 days', '90 days', '180 days', '365 days', '730 days', '1095 days',][last_iph['initial']],
    'period': ['45 days', '90 days', '180 days', '365 days', '730 days', '1095 days'][last_iph['period']],
    'horizon': ['45 days', '90 days', '180 days', '365 days', '730 days'][last_iph['horizon']]
}

print(best_iph)


# HİPERPARAMETRE OPTİMİZASYONU #

def objective1(params):
    model1 = Prophet(**params)
    model1.fit(df)  # Modeli eğitelim
    df_cv1 = cross_validation(model1, initial=best_iph["initial"], period=best_iph["period"],
                              horizon=best_iph["horizon"])
    df_p1 = performance_metrics(df_cv1)
    return -df_p1['rmse'].values[0]


hyperparam_space = {
    'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.5),
    'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 10.0),
    'holidays_prior_scale': hp.uniform('holidays_prior_scale', 0.01, 10.0),
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])}


trials1 = Trials()
best = fmin(objective1, hyperparam_space, algo=tpe.suggest, max_evals=10, trials=trials1)
best["seasonality_mode"] = "additive"

print(best)



###########################################################
# EN İYİ MODEL İLE TAHMİNLEME #
###########################################################

model_best = Prophet(**best)
model_best.fit(df)


future = model_best.make_future_dataframe(periods=730)
predicts = model_best.predict(future)


# SKORLAMA #

y_true = df["y"]
y_pred = predicts["yhat"][:-730]

r2 = r2_score(y_true, y_pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("R^2 (R-kare):", r2)
print("RMSE:", rmse)


###########################################################
# ÖZELLİK ÖNEMİ #
###########################################################

model_best.plot_components(predicts)
plt.show()

###########################################################
# GÖRSELLEŞTİRME #
###########################################################

# Tahminlenen kısmı gerçek değerlerden farklı olarak görselleştir
fig = go.Figure()

# Gerçek verileri ekle
fig.add_trace(go.Scatter(x=df.index, y=df['y'], mode='lines', name='Gerçek Veriler'))

# Tahminleri ekle
fig.add_trace(go.Scatter(x=predicts['ds'], y=predicts['yhat'], mode='lines', name='Tahminlenen Veriler'))

# İleriye dönük tahminleri vurgula
fig.add_trace(go.Scatter(x=predicts['ds'].iloc[-730:], y=predicts['yhat'].iloc[-730:], mode='lines', fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.3)', name='İleriye Dönük Tahminler'))

# Grafik düzenleme
fig.update_layout(title='Prophet Tahminleri', xaxis_title='Tarih', yaxis_title='Değer')

# Grafiği HTML dosyası olarak kaydet
fig.write_html("prophet_tahminleri.html")





