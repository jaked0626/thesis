import pandas as pd
import dask.dataframe as dd
import os
from dotenv import load_dotenv
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data() -> dd.DataFrame:
    load_dotenv()
    csv_dir = os.getenv("RAW_NTT_DATA_PATH")
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv') and f[0] != "."]
    col_names = ["date", "day_of_week", "time", "area", "residence", "age", "gender", "population"]
    print(f"loading following files into dataframe object: {csv_files}")
    ntt_df = dd.read_csv(csv_files, 
                         names=col_names,
                         dtype={'date': 'str', 'day_of_week': 'str', 'time': 'str', 'area': 'str',
                                'residence': 'str', 'age': 'Int64', 'gender': 'str', 'population': 'Int64'})
    ntt_df = ntt_df.drop(columns = ["residence", "age", "gender"])
    ntt_df = ntt_df.dropna()
    
    return ntt_df


def load_data_pandas() -> pd.DataFrame:
    load_dotenv()
    csv_dir = os.getenv("RAW_NTT_DATA_PATH")
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv') and f[0] != "." and f[6:10] != "2019"] # only interested in covid
    col_names = ["date", "day_of_week", "time", "area", "residence", "age", "gender", "population"]
    datatypes = {'date': 'str', 'day_of_week': 'str', 'time': 'str', 'area': 'str',
                 'residence': 'str', 'age': 'Int64', 'gender': 'str', 'population': 'Int64'}
    df_lst = []
    print(f"loading following files into dataframe object: {csv_files}")
    for filepath in csv_files:
        print(filepath)
        df = pd.read_csv(filepath, names = col_names, dtype = datatypes, on_bad_lines="skip")
        df_lst.append(df)
    ntt_df = pd.concat(df_lst, axis = 0)

    return ntt_df


def filter_area(area_name:str, area_prefixes: list[str]) -> pd.DataFrame:
    load_dotenv()
    csv_dir = os.getenv("RAW_NTT_DATA_PATH")
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv') and f[0] != "." and f[6:10] != "2019"]
    col_names = ["date", "day_of_week", "time", "area", "residence", "age", "gender", "population"]
    datatypes = {'date': 'str', 'day_of_week': 'str', 'time': 'str', 'area': 'str',
                 'residence': 'str', 'age': 'Int64', 'gender': 'str', 'population': 'Int64'}
    df_lst = []
    print(f"loading following files into dataframe object: {csv_files}")
    for i, filepath in enumerate(csv_files):
        print(f"parsing {filepath}")
        df = pd.read_csv(filepath, names = col_names, dtype = datatypes, on_bad_lines="skip")
        conditions = df['area'].str.startswith(area_prefixes[0])
        for prefix in area_prefixes[1:]:
            conditions |= df['area'].str.startswith(prefix)
        df = df[conditions]
        df_lst.append(df)
        print(f"{round((i + 1) * 100 / len(csv_files))}% complete")
    ntt_df = pd.concat(df_lst, axis = 0)
    ntt_df.to_csv(f"./data/ntt_data/{area_name}.csv")

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
                                   'area': 'str', 'residence': 'str', 'age': 'Int64', 
                                   'gender': 'str', 'population': 'Int64'})
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
    add_dummies(shibuya_daily_pop_covid)

    shibuya_daily_pop_covid.to_csv(f"{datapath}/ntt_data/shibuya_daily_pop_covid.csv", index=False)

    return shibuya_daily_pop_covid


def plot_shibuya_covid(day_of_the_week: str = False, log: bool = False):
    df = load_shibuya_daily_pop_covid()
    title = "Shibuya Station Population"

    if day_of_the_week:
        mask = df["date"].dt.day_name() == day_of_the_week
        df = df[mask]
        title += f" ({day_of_the_week}s)"

    if log:
        df = df
        df[['daily_avg_population', 'total_new_cases', 'total_severe_cases', 'total_daily_deaths']] = \
            df[['daily_avg_population', 'total_new_cases', 'total_severe_cases', 'total_daily_deaths']].apply(np.log)
        title += " (Logged)"

    # Extract the 'daily_avg_population', 'total_new_cases', 'total_severe_cases', 'total_daily_deaths'
    # columns as separate arrays
    dates = df['date'].values
    daily_avg_population = df['daily_avg_population'].values
    total_new_cases = df['total_new_cases'].values
    total_severe_cases = df['total_severe_cases'].values
    total_daily_deaths = df['total_daily_deaths'].values

    # Extract the 'soe1', 'soe2', 'soe3'... columns as separate arrays
    soe1 = df['soe1'].values
    soe2 = df['soe2'].values
    soe3 = df['soe3'].values
    soe4 = df['soe4'].values
    semisoe1 = df['semi-soe1'].values
    semisoe2 = df['semi-soe2'].values

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Plot the quantities of interest on the primary y-axis
    ax.plot(dates, daily_avg_population, color='blue', label='Daily Average Population')
    ax.plot(dates, total_new_cases, color='red', label='total_new_cases')
    ax.plot(dates, total_severe_cases, color='green', label='total_severe_cases')
    ax.plot(dates, total_daily_deaths, color='purple', label='total_daily_deaths')


    # Set the y-axis label
    ax.set_ylabel('Population')

    # show legend
    plt.legend()

    # Create a secondary y-axis with a different color map
    ax2 = ax.twinx()

    # Plot the dummy variables on the secondary y-axis with different colors
    ax2.plot(dates, soe1, color='red', label='SOE1')
    ax2.plot(dates, soe2, color='orange', label='SOE2')
    ax2.plot(dates, soe3, color='green', label='SOE3')
    ax2.plot(dates, soe4, color='purple', label='SOE4')
    ax2.plot(dates, semisoe1, color='pink', label='SEMI-SOE1')
    ax2.plot(dates, semisoe2, color='yellow', label='SEMI_SOE2')

    # Set the y-axis label and y-axis limits
    ax2.set_ylabel('SOE Status')
    ax2.set_ylim(0, 1)

    # Show the legend
    plt.legend()
    plt.title(title)
    # Show the plot
    plt.show()
    # shibuya_daily_pop_covid.plot(x='date', 
    #                              y=['daily_avg_population', 'total_new_cases', 'tokyo_new_cases', 'tokyo_severe_cases'])
    # plt.title(title)
    # plt.show()



def add_dummies(df: pd.DataFrame):
    """
    Modifies dataframe inplace by adding dummy variables to indicate state of emergency 
    and semi state of emergency. 
    Inputs:
        df: pandas DataFrame with 'date' column of the form YYYY-mm-dd
    """
    def create_dummy_variable(row, lower_bound, upper_bound):
        lower_bound = pd.to_datetime(lower_bound)
        upper_bound = pd.to_datetime(upper_bound)
        if lower_bound <= row['date'] <= upper_bound:
            return 1
        else:
            return 0

    # first state of emergency
    df['soe1'] = df.apply(create_dummy_variable, args=('2020-04-07', '2020-05-25'), axis=1)
    # second soe 
    df['soe2'] = df.apply(create_dummy_variable, args=('2021-1-8', '2021-3-21'), axis=1)
    # third soe 
    df['soe3'] = df.apply(create_dummy_variable, args=('2021-4-25', '2021-6-20'), axis=1)
    # fourth soe
    df['soe4'] = df.apply(create_dummy_variable, args=('2021-7-12', '2021-9-12'), axis=1)

    # first semi state of emergency
    df['semi-soe1'] = df.apply(create_dummy_variable, args=('2021-4-12', '2021-4-24'), axis=1)
    # second semi soe
    df['semi-soe2'] = df.apply(create_dummy_variable, args=('2021-6-21', '2021-7-11'), axis=1)

    # first wave 
    df['wave1'] = df.apply(create_dummy_variable, args=('2020-1-29', '2020-6-13'), axis=1)
    # second wave
    df['wave2'] = df.apply(create_dummy_variable, args=('2020-6-14', '2020-10-9'), axis=1)
    # third wave
    df['wave3'] = df.apply(create_dummy_variable, args=('2020-10-10', '2021-2-28'), axis=1)
    # fourth wave
    df['wave4'] = df.apply(create_dummy_variable, args=('2021-3-1', '2021-6-20'), axis=1)
    # fifth wave
    df['wave5'] = df.apply(create_dummy_variable, args=('2021-6-21', '2021-12-31'), axis=1)


def main():
    # load_shibuya_daily_pop_covid()
    filter_area("shibuya_station_1", ["53393595", "53393596", "53393585", "53393586"])
    filter_area("tokyo_ootemachi", ["53394611", "53394621"])
    filter_area("mitaka", ["53394434", "53394444"])


if __name__ == "__main__":
    main()

