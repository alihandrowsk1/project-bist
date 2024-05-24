###########################################################
# IMPORT'LAR VE BAZI AYARLAR #
###########################################################

import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import evds as e
from datetime import datetime, timedelta
from APIKEYS import key

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

###########################################################
# VERİ SETİ OLUŞTURMA #
###########################################################

# EVDS API ANAHTARI #
evds = e.evdsAPI(key)
"""
Burdaki "key" argümanı evds web sitesinden alınan ve evds küpühanesine 
bağlanmak için kişiye özel tanımlanmış bir anahtardır.

(EVDS SİSTEMİ ÜZERİNDEN BİR HESAP OLUŞTURUP HESABIN API KEY'İ İLE BU
ÇALIŞMANIN DOSYA DİZİNİNDE APIKEYS.py ADLI DOSYA OLUŞTURUP ALINAN KEY,
DOSYANIN İÇİNDE TANIMLANDIKTAN SONRA HERHANGİ BİR AYAR DEĞİŞTİRMEDEN
SORUNSUZ ÇALIŞTIRILABİLİR)
"""

print("MODEL BAŞLATILIYOR...")

# KUR SEPETİ VERİSİ #

exc = evds.get_data(["TP.DK.USD.S.YTL", "TP.DK.EUR.S.YTL"],
                    startdate="01-01-2018",
                    enddate="31-03-2024",
                    frequency=1, aggregation_types="last")

exc["KUR_SEPETİ"] = (exc["TP_DK_USD_S_YTL"] / 2) + (exc["TP_DK_EUR_S_YTL"] / 2)

# ENFLASYON VERİSİ #

enf = evds.get_data(["TP.FG.J0", "TP.TUFE1YI.T1"],
                    startdate="01-01-2018",
                    enddate="31-03-2024",
                    frequency=1)

enf.rename(columns={"TP_FG_J0": "TUFE", "TP_TUFE1YI_T1": "UFE"}, inplace=True)

daily_enf = pd.DataFrame(columns=["Tarih", "TUFE"])

for index, row in enf.iterrows():
    year, month = map(int, row["Tarih"].split("-"))
    days_in_month = (datetime(year, month % 12 + 1, 1) - timedelta(days=1)).day
    daily_increase_tufe = (row["TUFE"] - enf.iloc[index - 1]["TUFE"]) / days_in_month
    daily_data = []
    for day in range(1, days_in_month + 1):
        date = f"{year}-{month:02d}-{day:02d}"
        daily_tufe = row["TUFE"] - daily_increase_tufe * (days_in_month - day)
        daily_data.append({"Tarih": date, "TUFE": daily_tufe})
    daily_enf = pd.concat([daily_enf, pd.DataFrame(daily_data)])

"""
AŞAĞIDAKİ BLOK REPO YENİDEN TARİHLEME İŞLEMİ İÇİN KULLANILIYOR.
"""
daily_enf["Tarih"] = pd.to_datetime(daily_enf["Tarih"], format="%Y-%m-%d")

min_tarih = daily_enf["Tarih"].min()
max_tarih = daily_enf["Tarih"].max()

# REPO VERİSİ #

rep0 = evds.get_data(["TP.AOFOBAP"],
                     startdate="01-01-2018",
                     enddate="31-03-2024",
                     frequency=1)

rep0["Tarih"] = pd.to_datetime(rep0["Tarih"], format='%d-%m-%Y')
rep0.set_index("Tarih", inplace=True)

tum_tarihler = pd.date_range(start=min_tarih, end=max_tarih)
tam_df = pd.DataFrame({"Tarih": tum_tarihler})
repo = pd.merge(tam_df, rep0, on="Tarih", how='left')

repo.rename(columns={"TP_AOFOBAP": "REPO"}, inplace=True)

# YABANCI, YERLİ TAKAS VERİSİ #

takas = pd.read_excel("takas_verisi.xlsx", index_col="TARIH")

daily_takas = takas.resample('D').asfreq()
daily_takas = daily_takas[:-1]
daily_takas = daily_takas.interpolate(method='linear')
daily_takas.columns = pd.MultiIndex.from_tuples([('TAKAS', col) for col in daily_takas.columns])
daily_takas = daily_takas[: -30]


###########################################################
# VERİ SETLERİNİN BİRLEŞTİRİLMESİ VE DÜZENLENMESİ #
###########################################################

def indir_ve_return_et(n):
    data_sets = []
    stock_codes = []

    for i in range(n):
        y_stock = input(f"{i + 1}. TAHMİNLENECEK HİSSE SENEDİ'NİN KODU NEDİR?").upper()
        stock_codes.append(y_stock)
        stocks = [f"{y_stock}.IS", "XU100.IS", "BZ=F", "GC=F", "BTC-USD"]

        print(f"{i + 1}. Veri seti indiriliyor...")
        data = yf.download(stocks, start="2018-01-01", end="2024-04-01", group_by="ticker")

        # Sadece belirli kolonları seçelim
        selected_columns = [
            (f"{y_stock}.IS", "Open"),
            (f"{y_stock}.IS", "High"),
            (f"{y_stock}.IS", "Low"),
            (f"{y_stock}.IS", "Close"),
            (f"{y_stock}.IS", "Volume"),
            ("XU100.IS", "Open"),
            ("XU100.IS", "High"),
            ("XU100.IS", "Low"),
            ("XU100.IS", "Close"),
            ("XU100.IS", "Volume"),
            ("BZ=F", "Open"),
            ("BZ=F", "High"),
            ("BZ=F", "Low"),
            ("BZ=F", "Close"),
            ("BZ=F", "Volume"),
            ("GC=F", "Open"),
            ("GC=F", "High"),
            ("GC=F", "Low"),
            ("GC=F", "Close"),
            ("GC=F", "Volume"),
            ("BTC-USD", "Open"),
            ("BTC-USD", "High"),
            ("BTC-USD", "Low"),
            ("BTC-USD", "Close"),
            ("BTC-USD", "Volume")
        ]

        data = data[selected_columns]

        # Ek veri setlerinin eklenmesi
        data[("OTHER_VALUES", "EXCHANGE_BASKET")] = exc["KUR_SEPETİ"].values
        data[("OTHER_VALUES", "TUFE")] = daily_enf["TUFE"].values
        data[("OTHER_VALUES", "REPO")] = repo["REPO"].values
        data = pd.concat([data, daily_takas], axis=1)

        # Verileri temizleme
        df_clean = data.dropna(subset=[(f"{y_stock}.IS", "Close")])
        df_clean.fillna(method="ffill", inplace=True)
        df_clean.fillna(method="bfill", inplace=True)

        # LAG sütunlarının eklenmesi
        for j in range(1, 15):
            yeni_sutun = f"LAG_{j}"
            df_clean[("LAGS", yeni_sutun)] = df_clean[(f"{y_stock}.IS", "Close")].shift(j)

        # RSI hesaplaması ve eklenmesi
        window = 14
        close_prices = df_clean[(f"{y_stock}.IS", "Close")]
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df_clean[("INDICATORS", "RSI")] = rsi

        # Kolon isimlerini düzeltme
        df_clean.columns = ['_'.join(col) for col in df_clean.columns]

        data_sets.append(df_clean)
        print("VERİLER İNDİRİLDİ.")

    dfs = {stock_codes[i]: data_sets[i] for i in range(n)}

    return dfs, data_sets, stock_codes


dfs, data_sets, stock_codes = indir_ve_return_et(5)
