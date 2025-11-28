import pandas as pd
import glob
import os
import holidays

data_folder = "/Users/mcwon/Documents/수학통계/Data_Collecting/Data_Collect"
output_folder = "../data"
os.makedirs(output_folder, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
all_hourly = []

for file_path in csv_files:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    
    if '일시' not in df.columns:
        raise KeyError(f"'일시' 컬럼이 CSV 파일에 없습니다: {file_path}")
    
    df['일시'] = pd.to_datetime(df['일시'])
    df = df.set_index('일시')
    
    df_hourly = df.resample('h').mean()
    
    if '현재부하(MW)' not in df_hourly.columns:
        raise KeyError(f"'현재부하(MW)' 컬럼이 CSV 파일에 없습니다: {file_path}")
    
    df_hourly = df_hourly[['현재부하(MW)']].rename(columns={'현재부하(MW)': 'load'})
    all_hourly.append(df_hourly)

full_data = pd.concat(all_hourly).sort_index()

wed_data_folder = "/Users/mcwon/Documents/수학통계/Data_Collecting/weather_DATA"
weather_file = os.path.join(wed_data_folder, "weather_final.csv")
weather = pd.read_csv(weather_file, encoding="utf-8-sig")

weather.columns = weather.columns.str.strip()
if '일시' not in weather.columns:
    raise KeyError("'일시' 컬럼이 날씨 CSV에 없습니다.")

weather['일시'] = pd.to_datetime(weather['일시'])
weather = weather.set_index('일시')
weather = weather[['기온(°C)', '습도(%)']].rename(columns={'기온(°C)': 'temp', '습도(%)': 'humidity'})


full_data = full_data.merge(weather, left_index=True, right_index=True, how='inner')
kr_holidays = holidays.KR(years=range(full_data.index.min().year, full_data.index.max().year + 1))

window_size = 24
rows = []

for i in range(window_size, len(full_data)):
    past_load = full_data['load'].iloc[i-window_size:i].values.tolist()
    
    next_dt = full_data.index[i]
    year = next_dt.year
    hour = next_dt.hour
    day_of_week = next_dt.dayofweek
    month = next_dt.month
    is_holiday = 1 if next_dt.date() in kr_holidays else 0
    
    next_load = full_data['load'].iloc[i]
    temp = full_data['temp'].iloc[i]
    humidity = full_data['humidity'].iloc[i]
    
    rows.append(past_load + [hour, day_of_week, month, is_holiday, temp, humidity, next_load, year])

columns = (
    [f'past_load_{i}' for i in range(window_size)]
    + ['hour', 'day_of_week', 'month', 'is_holiday', 'temp', 'humidity', 'next_load', 'year']
)

df_model = pd.DataFrame(rows, columns=columns)
df_model = pd.get_dummies(df_model, columns=['day_of_week'], prefix='dow')

output_path = os.path.join(output_folder, "epsis_data_2019_2024.csv")
df_model.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"학습용 CSV 생성 완료: {output_path}")

