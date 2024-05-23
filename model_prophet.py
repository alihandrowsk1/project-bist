###########################################################
# IMPORT'LAR VE BAZI AYARLAR #
###########################################################

import warnings
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import hp, fmin, tpe, Trials
from veri_seti import data_sets, stock_codes


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


###########################################################
# MODELLEME #
###########################################################


def prepare_data_for_prophet(dataframes, stock_codes):
    # Veri setlerini Prophet formatına uygun hale getirir
    processed_dataframes = []
    scalers = []

    for df, stock_code in zip(dataframes, stock_codes):
        ys = f"{stock_code}.IS_Close"
        y_df = df[[ys]]  # "y" değişkeni için ayrı DataFrame
        X_df = df.drop([ys], axis=1)

        # "y" değişkeni için ölçeklendirme işlemi
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y_df)
        y_df_scaled = pd.DataFrame(y_scaled, columns=[ys], index=y_df.index)

        # Diğer değişkenler için ölçeklendirme işlemi
        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X_df)
        X_df_scaled = pd.DataFrame(X_scaled, columns=X_df.columns, index=X_df.index)

        # 'ds' ve 'y' sütunlarını ekleyen fonksiyonun çağrılması
        X_df_scaled['ds'] = X_df_scaled.index
        X_df_scaled['y'] = y_df_scaled.values

        # DataFrame'i birleştirme
        processed_dataframes.append(X_df_scaled)
        scalers.append((y_scaler, X_scaler))

    return processed_dataframes, scalers


df, scalers = prepare_data_for_prophet(data_sets, stock_codes)

original_df = data_sets.copy()  # ölçeklenmemiş veri seti

# BAĞIMLI DEĞİŞKENLERİN ÖLÇEKLEYİCİLERİNİ LİSTELEME #
sca = np.array(scalers)
array_scaler = sca[:, 0]
y_scalers = list(array_scaler)


df_hyper = df[0]  # optimizasyon için kullanılacak df


###########################################################
# OPTİMAL HİPERPARAMETRE AYARLARI #
###########################################################

def objective_model_params(params):
    model = Prophet(**params)
    model.fit(df_hyper)
    df_cv = cross_validation(model, initial="730 days", period="365 days", horizon="365 days")
    df_performance = performance_metrics(df_cv)
    return df_performance['rmse'].values[0]


hyperparam_space = {
    'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.001, 0.5),
    'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 10.0),
    'holidays_prior_scale': hp.uniform('holidays_prior_scale', 0.01, 10.0),
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
}

trials_model = Trials()
best_params = fmin(objective_model_params, hyperparam_space, algo=tpe.suggest, max_evals=10, trials=trials_model)
best_params["seasonality_mode"] = "additive"

print(best_params)


###########################################################
# EN İYİ MODEL İLE TAHMİNLEME #
###########################################################

while True:
    try:
        periods = int(input("KAÇ GÜN TAHMİNLEMEK İSTİYORSUNUZ? "))
        break
    except ValueError:
        print("Lütfen geçerli bir sayı girin.")


def predict_with_best_model(dataframes, periods):
    all_predictions = []
    for df in dataframes:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            holidays_prior_scale=best_params['holidays_prior_scale'],
            seasonality_mode=best_params['seasonality_mode'])
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        predictions = model.predict(future)
        all_predictions.append(predictions)
    return all_predictions


predictions = predict_with_best_model(df, periods)


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


# ÖLÇEKLENMİŞ TAHMİNLENEN DEĞERLERİ ORİJİNAL HALİNE DÖNDÜRME


def inverse_transform_predictions(all_predictions, scalers):
    inverted_predictions = []
    for predictions, scaler in zip(all_predictions, scalers):
        # Identify columns that should not be transformed
        non_transform_columns = ['ds']
        # Separate the columns to be transformed and those that should not be transformed
        columns_to_transform = [col for col in predictions.columns if col not in non_transform_columns]

        # Apply inverse transform to the selected columns
        transformed_values = scaler.inverse_transform(predictions[columns_to_transform])

        # Create a new DataFrame for transformed values and add non-transformed columns back
        transformed_df = pd.DataFrame(transformed_values, columns=columns_to_transform)
        for col in non_transform_columns:
            transformed_df[col] = predictions[col].values

        # Reorder the columns to match the original DataFrame
        transformed_df = transformed_df[predictions.columns]

        inverted_predictions.append(transformed_df)
    return inverted_predictions


predictions_nonscale = inverse_transform_predictions(predictions, y_scalers)


###########################################################
# GÖRSELLEŞTİRME #
###########################################################

def visualize_predictions(df_list, stock_codes, predictions):
    for i, (dataf, stock_code, preds) in enumerate(zip(df_list, stock_codes, predictions)):
        plt.figure(figsize=(14, 8))
        plt.plot(dataf.index, dataf[f"{stock_code}.IS_Close"], label="Gerçek Veriler", color="blue")
        plt.plot(preds['ds'], preds['yhat'], label="Tahminler", color="red")
        plt.fill_between(preds['ds'], preds['yhat_lower'], preds['yhat_upper'], color="pink", alpha=0.3)

        # Son değeri belirle ve metin olarak ekler
        last_real_value = dataf[f"{stock_code}.IS_Close"].iloc[-1]
        last_pred_value = preds['yhat'].iloc[-1]
        plt.text(dataf.index[-1], last_real_value, f"{last_real_value:.2f}",
                 verticalalignment='bottom', horizontalalignment='right', color='blue')
        plt.text(preds['ds'].iloc[-1], last_pred_value, f"{last_pred_value:.2f}",
                 verticalalignment='bottom', horizontalalignment='right', color='red')

        plt.xlabel("Tarih")
        plt.ylabel("Fiyat")
        plt.title(f"{stock_code} için Gerçek Veriler ve Tahminler")
        plt.legend()
        plt.grid(True)
        plt.show()


visualize_predictions(original_df, stock_codes, predictions_nonscale)


###########################################################
# PIPELINE #
###########################################################

def main():
    print("BÜTÜN İŞLEMLER BAŞARIYLA GERÇEKLEŞTİRİLDİ.")


if __name__ == '__main__':
    main()
