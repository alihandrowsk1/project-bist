###########################################################
# IMPORT'LAR VE BAZI AYARLAR #
###########################################################

import warnings
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from hyperopt import hp, fmin, tpe, Trials
from veri_seti import dfs, data_sets, stock_codes

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


###########################################################
# MODELLEME #
###########################################################

# MODEL İÇİN VERİ SETİNİ UYGUN HALE GETİRME #

def dataf_for_prophet(dataframes, stock_codes):
    processed_dataframes = []

    for df_ds, stock_code in zip(dataframes, stock_codes):
        ys = f"{stock_code}.IS_Close"
        df_ds = df_ds.copy()
        df_ds["ds"] = df_ds.index
        df_ds["y"] = df_ds[ys]
        df_ds.drop([ys], axis=1, inplace=True)
        processed_dataframes.append(df_ds)

    return processed_dataframes


df = dataf_for_prophet(data_sets, stock_codes)
first_df = df[0]  # optimizasyon için kullanılacak


###########################################################
# OPTİMAL HİPERPARAMETRE AYARLARI #
###########################################################

def objective0(params):
    initial = params['initial']
    period = params['period']
    horizon = params['horizon']

    model0 = Prophet()
    model0.fit(first_df)
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
    'initial': ['45 days', '90 days', '180 days', '365 days', '730 days', '1095 days', ][last_iph['initial']],
    'period': ['45 days', '90 days', '180 days', '365 days', '730 days', '1095 days'][last_iph['period']],
    'horizon': ['45 days', '90 days', '180 days', '365 days', '730 days'][last_iph['horizon']]
}

print(best_iph)


# HİPERPARAMETRE OPTİMİZASYONU #

def objective1(params):
    model1 = Prophet(**params)
    model1.fit(first_df)  # Modeli eğitelim
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
periods = int(input("KAÇ GÜN TAHMİNLEMEK İSTİYORSUNUZ?"))


def last_model(dataframes, periods):
    all_predicts = []
    for dataf in dataframes:
        model = Prophet(**best)
        model.fit(dataf)
        future = model.make_future_dataframe(periods=periods)
        predicts = model.predict(future)
        all_predicts.append(predicts)
    return all_predicts


predictions = last_model(df, periods)

# SKORLAMA #
def calculate_scores(df_list, predictions):
    rmse_scores = []
    r2_scores = []

    for i, (df, preds) in enumerate(zip(df_list, predictions)):
        min_len = min(len(df), len(preds))
        y_true = df['y'].values[:min_len]
        y_pred = preds['yhat'].values[:min_len]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        r2 = r2_score(y_true, y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"Model {i + 1} için RMSE: {rmse}")
        print(f"Model {i + 1} için R^2: {r2}")

    return rmse_scores, r2_scores


rmse_scores, r2_scores = calculate_scores(df, predictions)


###########################################################
# GÖRSELLEŞTİRME #
###########################################################

def visualize_predictions(df_list, stock_codes, predictions):
    for i, (dataf, stock_code, preds) in enumerate(zip(df_list, stock_codes, predictions)):
        plt.figure(figsize=(10, 6))
        plt.plot(dataf['ds'], dataf['y'], label="Gerçek Veriler", color="blue")
        plt.plot(preds['ds'], preds['yhat'], label="Tahminler", color="red")
        plt.fill_between(preds['ds'], preds['yhat_lower'], preds['yhat_upper'], color="pink", alpha=0.5)
        plt.xlabel("Tarih")
        plt.ylabel("Değer")
        plt.title(f"{stock_code} için Gerçek Veriler ve Tahminler")
        plt.legend()
        plt.grid(True)
        plt.show()


visualize_predictions(df, stock_codes, predictions)


###########################################################
# PIPELINE #
###########################################################

def main():
    print("BÜTÜN İŞLEMLER BAŞARIYLA GERÇEKLEŞTİRİLDİ.")


if __name__ == '__main__':
    main()
