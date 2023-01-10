import pandas as pd
import dask.dataframe as dd
import os
from dotenv import load_dotenv
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_data() -> dd.DataFrame:
    load_dotenv()
    csv_dir = os.getenv("RAW_NTT_DATA_PATH")
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv') and f[0] != "."]
    col_names = ["date", "day_of_week", "time", "area", "residence", "age", "gender", "population"]
    print(f"loading following files into dataframe object: {csv_files}")
    ntt_df = dd.read_csv(csv_files, 
                         names=col_names,
                         dtype={'date': 'str', 'day_of_week': 'str', 'time': 'str', 'area': 'str',
                         'residence': 'str', 'age': 'int64', 'gender': 'str', 'population': 'int64'})
    ntt_df = ntt_df.drop(columns = ["residence", "age", "gender"])
    ntt_df = ntt_df.dropna()
    
    return ntt_df


def load_data_pandas() -> pd.DataFrame:
    load_dotenv()
    csv_dir = os.getenv("RAW_NTT_DATA_PATH")
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv') and f[0] != "."]
    col_names = ["date", "day_of_week", "time", "area", "residence", "age", "gender", "population"]
    datatypes = {'date': 'str', 'day_of_week': 'str', 'time': 'str', 'area': 'str',
                 'residence': 'str', 'age': 'int64', 'gender': 'str', 'population': 'int64'}
    df_lst = []
    print(f"loading following files into dataframe object: {csv_files}")
    for filepath in csv_files:
        print(filepath)
        df = pd.read_csv(filepath, names = col_names, dtype = datatypes)
        df_lst.append(df)
    ntt_df = pd.concat(df_lst, axis = 0)

    return ntt_df


def filter_area(area: str) -> pd.DataFrame:
    load_dotenv()
    csv_dir = os.getenv("RAW_NTT_DATA_PATH")
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv') and f[0] != "."]
    col_names = ["date", "day_of_week", "time", "area", "residence", "age", "gender", "population"]
    datatypes = {'date': 'str', 'day_of_week': 'str', 'time': 'str', 'area': 'str',
                 'residence': 'str', 'age': 'int64', 'gender': 'str', 'population': 'int64'}
    df_lst = []
    print(f"loading following files into dataframe object: {csv_files}")
    for filepath in csv_files:
        print(filepath)
        df = pd.read_csv(filepath, names = col_names, dtype = datatypes)
        df = df[df.area.str.startswith(area)]
        df_lst.append(df)
    ntt_df = pd.concat(df_lst, axis = 0)

    return ntt_df


def get_summary():
    files = [f for f in os.listdir(os.getenv("RAW_NTT_DATA_PATH")) if f.endswith("csv") and f[0] != "."]
    for f in files:
        print(f)
        col_names = ["date", "day_of_week", "time", "area", "residence", "age", "gender", "population"]
        filepath = os.path.join(os.getenv("RAW_NTT_DATA_PATH"), f)
        try:
            df = pd.read_csv(filepath, 
                             names = col_names, 
                             dtype={'date': 'str', 'day_of_week': 'str', 'time': 'str', 
                                   'area': 'str', 'residence': 'str', 'age': 'int64', 
                                   'gender': 'str', 'population': 'int64'})
            print(df.describe())
            print(df.population.mean())
        except Exception:
            print(Exception)
            print(f"{f} failed to read")


def process_weather_data():
    datapath = "./data/weather_data"
    # load dataframe
    weather_df = pd.concat(map(pd.read_csv, glob.glob(datapath + "/data*.csv")), ignore_index=True)
    # split datetime into date and time
    weather_df[["date", "time"]] = weather_df["datetime"].str.split(" ", expand=True)
    # format date col
    weather_df["date"] = pd.to_datetime(weather_df["date"], format="%Y/%m/%d")
    #weather_df["date"] = weather_df["date"].dt.strftime("%Y%m%d")
    # format time col
    weather_df["time"] = weather_df["time"].str.split(":", expand=True)[0] + weather_df["time"].str.split(":", expand=True)[1]
    weather_df["time"] = weather_df["time"].str.zfill(4)
    # sort and drop datetime
    weather_df["datetime"] = weather_df["date"] + weather_df["time"]
    weather_df.sort_values("datetime", inplace=True)
    weather_df.reset_index(drop=True, inplace=True)
    # fill na values in weather and convert to int
    weather_df["weather"] = weather_df["weather"].fillna(method='ffill')
    weather_df['weather'] = weather_df['weather'].replace(np.nan, 2.0)
    weather_df['weather'] = weather_df['weather'].astype(int)
    weather_df.drop("datetime", inplace=True, axis=1)
    print(weather_df)
    weather_df.to_csv(f"{datapath}/weather.csv", index=False)

    return weather_df


def load_weather_data():
    datapath = './data/weather_data'
    df = pd.read_csv(f"{datapath}/weather.csv", dtype={"date": str, "time": str})
    return df


def process_japan_covid_data():
    datapath = "./data/covid_data"
    new_cases_df = pd.read_csv(f"{datapath}/japan_new_cases_daily.csv")
    new_cases_df = new_cases_df[["Date", "ALL", "Tokyo"]]
    new_cases_df["Date"] = pd.to_datetime(new_cases_df["Date"], format="%Y/%m/%d")
    new_cases_df["Date"] = new_cases_df["Date"].dt.strftime("%Y%m%d")
    new_cases_df.rename(columns={"Date": "date", "ALL": "total_new_cases", "Tokyo": "tokyo_new_cases"}, inplace=True)
    new_cases_df["total_weekly_diff"] = new_cases_df["total_new_cases"] - new_cases_df["total_new_cases"].shift(7, fill_value = 0)
    new_cases_df["tokyo_weekly_diff"] = new_cases_df["tokyo_new_cases"] - new_cases_df["tokyo_new_cases"].shift(7, fill_value = 0)

    severe_cases_df = pd.read_csv(f"{datapath}/japan_severe_cases_daily.csv")
    severe_cases_df = severe_cases_df[["Date", "ALL", "Tokyo"]]
    severe_cases_df["Date"] = pd.to_datetime(severe_cases_df["Date"], format="%Y/%m/%d")
    severe_cases_df["Date"] = severe_cases_df["Date"].dt.strftime("%Y%m%d")
    severe_cases_df.rename(columns={"Date": "date", "ALL": "total_severe_cases", "Tokyo": "tokyo_severe_cases"}, inplace=True)

    deaths_df = pd.read_csv(f"{datapath}/japan_deaths_cumulative_daily.csv")
    deaths_df = deaths_df[["Date", "ALL", "Tokyo"]]
    deaths_df["Date"] = pd.to_datetime(deaths_df["Date"], format="%Y/%m/%d")
    deaths_df["Date"] = deaths_df["Date"].dt.strftime("%Y%m%d")
    deaths_df.rename(columns={"Date": "date", "ALL": "total_cum_deaths", "Tokyo": "tokyo_cum_deaths"}, inplace=True)
    deaths_df["total_daily_deaths"] = deaths_df["total_cum_deaths"] - deaths_df["total_cum_deaths"].shift(1, fill_value = 0)
    deaths_df["tokyo_daily_deaths"] = deaths_df["tokyo_cum_deaths"] - deaths_df["tokyo_cum_deaths"].shift(1, fill_value = 0)
    
    jp_covid_data = new_cases_df.merge(severe_cases_df, on="date", how="left")
    jp_covid_data = jp_covid_data.merge(deaths_df, on="date", how="left")

    jp_covid_data.to_csv(f"{datapath}/jp_covid_data.csv", index=False)

    return jp_covid_data


def load_shibuya_daily_pop_covid():
    datapath = "./data"
    df = pd.read_csv(f"{datapath}/ntt_data/shibuya_station.csv")
    df_grouped = df.groupby(['date', 'time'])
    df_agg = df_grouped['population'].sum()
    df_daily_mean = df_agg.groupby("date").mean("population")
    df_daily_mean = df_daily_mean.to_frame(name="daily_avg_population")
    df_daily_mean = df_daily_mean.sort_values(by='date')
    df_mean_cov = df_daily_mean.loc[20200101:]

    jp_covid_data = pd.read_csv(f"{datapath}/covid_data/jp_covid_data.csv")
    shibuya_daily_pop_covid = df_mean_cov.merge(jp_covid_data, on="date") # how="left"? 
    shibuya_daily_pop_covid["date"] = pd.to_datetime(shibuya_daily_pop_covid["date"], format="%Y%m%d")

    shibuya_daily_pop_covid.to_csv(f"{datapath}/ntt_data/shibuya_daily_pop_covid.csv")

    return shibuya_daily_pop_covid


def plot_shibuya_covid(day_of_the_week: str = False, log: bool = False):
    shibuya_daily_pop_covid = load_shibuya_daily_pop_covid()
    title = "Shibuya Station Population"

    if day_of_the_week:
        mask = shibuya_daily_pop_covid["date"].dt.day_name() == day_of_the_week
        shibuya_daily_pop_covid = shibuya_daily_pop_covid[mask]
        title += f" ({day_of_the_week}s)"

    if log:
        shibuya_daily_pop_covid_log = shibuya_daily_pop_covid
        shibuya_daily_pop_covid_log[['daily_avg_population', 'total_new_cases', 'tokyo_new_cases', 'tokyo_severe_cases']] = \
            shibuya_daily_pop_covid[['daily_avg_population', 'total_new_cases', 'tokyo_new_cases', 'tokyo_severe_cases']].apply(np.log)
        title += " (Logged)"

    shibuya_daily_pop_covid.plot(x='date', 
                                 y=['daily_avg_population', 'total_new_cases', 'tokyo_new_cases', 'tokyo_severe_cases'])
    plt.title(title)
    plt.show()





