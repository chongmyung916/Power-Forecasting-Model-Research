import pandas as pd

file_path = "/Users/mcwon/Downloads/OBS_ASOS_TIM_20251013175004.csv"
df = pd.read_csv(file_path, encoding="cp949")
new_dates = pd.date_range(start="2019-01-01 00:00", end="2019-12-31 23:00", freq="H")
print(len(new_dates), len(df))

df["일시"] = new_dates
output_path = "/Users/mcwon/Documents/수학통계/Data_Collecting/weather_DATA/2019_weather_data.csv"
df.to_csv("/Users/mcwon/Documents/수학통계/Data_Collecting/weather_DATA/2019_weather_data.csv", 
          index=False, encoding="utf-8-sig")
