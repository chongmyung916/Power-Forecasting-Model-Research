import os
import pandas as pd

folder_path = "/Users/mcwon/Documents/수학통계/Data_Collecting/weather_DATA"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
dataframes = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    print(f"Loading {file_path} ...")
    df = pd.read_csv(file_path)
    df["일시"] = pd.to_datetime(df["일시"])
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)
merged_df = merged_df.sort_values(by="일시")
merged_df = merged_df.reset_index(drop=True)

output_path = os.path.join(folder_path, "weather_final.csv")
merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nClear weather merge!: {output_path}")