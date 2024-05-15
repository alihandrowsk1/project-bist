###########################################################
# IMPORT'LAR VE BAZI AYARLAR #
###########################################################

import yfinance as yf
import warnings
import pandas as pd
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
SORUNSUZ ÇALIŞTIRILABİLİR.)
"""


# KUR SEPETİ VERİSİ #

exc = evds.get_data(["TP.DK.USD.S.YTL", "TP.DK.EUR.S.YTL"],
                    startdate="01-01-2018",
                    enddate="31-12-2023",
                    frequency=1, aggregation_types="last")

exc["KUR_SEPETİ"] = (exc["TP_DK_USD_S_YTL"] / 2) + (exc["TP_DK_EUR_S_YTL"] / 2)


# ENFLASYON VERİSİ #

enf = evds.get_data(["TP.FG.J0", "TP.TUFE1YI.T1"],
                    startdate="01-01-2018",
                    enddate="31-12-2023",
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
                     enddate="31-12-2023",
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


# HİSSE, BİST100 VB. BORSA VERİLERİ #

stocks = ["KCHOL.IS", "XU100.IS", "BZ=F", "GC=F", "BTC-USD"]

data = yf.download(stocks, start="2018-01-01", end="2024-01-01", group_by="ticker")


###########################################################
# VERİ SETİNİN BİRLEŞTİRİLMESİ VE DÜZENLENMESİ #
###########################################################

data[("OTHER_VALUES", "EXCHANGE_BASKET")] = exc["KUR_SEPETİ"].values
data[("OTHER_VALUES", "TUFE")] = daily_enf["TUFE"].values
data[("OTHER_VALUES", "REPO")] = repo["REPO"].values
data = pd.concat([data, daily_takas], axis=1)

df_clean = data.dropna(subset=[("XU100.IS", "Close")])

df_clean.fillna(method="ffill", inplace=True)
df_clean.fillna(method="bfill", inplace=True)


# BAĞIMLI DEĞİŞKENİN GECİKMELİ DEĞERLERİ #

for i in range(1, 8):
    yeni_sutun = f"LAG_{i}"
    df_clean[("LAGS", yeni_sutun)] = df_clean[("KCHOL.IS", "Close")].shift(i)


###########################################################
# KEŞİFÇİ VERİ ANALİZİ #
###########################################################

"""
 DESCRIBE EKLENECEK
 BAĞIMSIZ DEĞİŞKEN KORELASYON EKLENECEK
 PAIRPLOT EKLENEBİLİR
"""
