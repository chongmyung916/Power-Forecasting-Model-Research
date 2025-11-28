import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import random
import shap
from statsmodels.graphics.tsaplots import plot_acf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(64)

base_output_dir = "/home/r21900053/stat_project/model_output"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
experiment_name = input("실험 이름을 입력하세요 (예: LSTM_GRU_GRID): ").strip() or "LSTM_GRU_GRID"
output_dir = os.path.join(base_output_dir, f"{timestamp}_{experiment_name}")
os.makedirs(output_dir, exist_ok=True)
print(f"실험 결과 저장 경로: {output_dir}")

data_path = "/home/r21900053/stat_project/Data_Collecting/Data_Final/epsis_data_2019_2024.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

df = pd.read_csv(data_path, encoding="utf-8-sig")
if '일시' not in df.columns:
    df['일시'] = pd.date_range(start="2019-01-01 00:00", periods=len(df), freq='H')

df['일시'] = pd.to_datetime(df['일시'])
df['year'] = df['일시'].dt.year

print("데이터 로우 수:", len(df))
print("컬럼들:", df.columns.tolist())

print("데이터 기본 통계량")
print(df.describe(include='all').transpose().head(20))
print("\n결측치 개수 (컬럼별)")
print(df.isna().sum())

binary_cols = ['is_holiday'] + [c for c in df.columns if c.startswith('dow_')]
numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(binary_cols)
if len(numeric_cols) == 0:
    raise ValueError("숫자형 컬럼이 없습니다. 데이터 확인 요망.")

try:
    z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
    outlier_mask = (z_scores > 3)
    print("\n이상치 개수 (Z-score > 3) per numeric column")
    print(outlier_mask.sum(axis=0))
except Exception as e:
    print("Z-score 계산 오류:", e)

df['temp_rate'] = df['temp'].diff()
df['temp_rate'] = df['temp_rate'].fillna(0)
print("temp rate 추가 완료")

method_choice = input("보간 방법을 선택하세요 ('linear' 또는 'spline') (default 'linear'): ").strip().lower()
if method_choice == "spline":
    df = df.interpolate(method="spline", order=2)
else:
    df = df.interpolate(method="linear")
df = df.ffill().bfill()
print("결측치 보간 완료")


apply_outlier_removal = input("이상치 제거(z-score>3)를 적용하시겠습니까? (y/N): ").strip().lower() == 'y'
if apply_outlier_removal:
    z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
    mask_keep = (z_scores < 3).all(axis=1)
    old_len = len(df)
    df = df[mask_keep].reset_index(drop=True)
    print(f"이상치 제거 완료: {old_len} -> {len(df)} rows")
else:
    print("이상치 제거는 수행하지 않았습니다.")


past_load_cols = [f"past_load_{i}" for i in range(24)]
static_cols = ['hour', 'month', 'is_holiday', 'temp', 'humidity', 'temp_rate'] + [c for c in df.columns if c.startswith("dow_")]
target_col = "next_load"

missing = [c for c in past_load_cols + static_cols + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"다음 컬럼들이 데이터에 없습니다: {missing}")

X_past = df[past_load_cols].values  
X_static = df[static_cols].values 
X = np.hstack([X_past, X_static])
y = df[target_col].values.reshape(-1, 1)
time_index = df['일시']

train_mask = df['year'].isin([2019, 2020, 2021, 2022])
val_mask   = df['year'] == 2023
test_mask  = df['year'] == 2024

X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
X_past_train, X_past_val, X_past_test = X_past[train_mask], X_past[val_mask], X_past[test_mask]
X_static_train, X_static_val, X_static_test = X_static[train_mask], X_static[val_mask], X_static[test_mask]
y_train, y_val, y_test = y[train_mask].flatten(), y[val_mask].flatten(), y[test_mask].flatten()
time_test = time_index[test_mask].reset_index(drop=True)

print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))
print(np.isnan(X_train).sum())
print(np.isinf(X_train).sum())

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
y_val = scaler_y.transform(y_val.reshape(-1,1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1,1)).flatten()

class LoadDataset(Dataset):
    def __init__(self, X_full, target):
        past = X_full[:, :24].reshape(-1, 24, 1)
        static = X_full[:, 24:]
        self.past = torch.tensor(past, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
    def __len__(self):
        return len(self.past)
    def __getitem__(self, idx):
        return self.past[idx], self.static[idx], self.target[idx]


class LSTMForecast(nn.Module):
    def __init__(self, static_dim, hidden_dim=64, static_embed_dim=32, dropout=0.2, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers ,batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.fc_static = nn.Linear(static_dim, static_embed_dim)
        self.fc_out = nn.Linear(hidden_dim + static_embed_dim, 1)
    def forward(self, past_seq, static_feat):
        
        _, (h_n, _) = self.lstm(past_seq) 
        h_n = h_n[-1]
        static_out = torch.relu(self.fc_static(static_feat))
        combined = torch.cat([h_n, static_out], dim=1)
        return self.fc_out(combined).squeeze(1)

class GRUForecast(nn.Module):
    def __init__(self, static_dim, hidden_dim=64, static_embed_dim=32, dropout=0.2, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, num_layers= num_layers, batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0)
        self.fc_static = nn.Linear(static_dim, static_embed_dim)
        self.fc_out = nn.Linear(hidden_dim + static_embed_dim, 1)
    def forward(self, past_seq, static_feat):
        _, h_n = self.gru(past_seq)
        h_n = h_n[-1]
        static_out = torch.relu(self.fc_static(static_feat))
        combined = torch.cat([h_n, static_out], dim=1)
        return self.fc_out(combined).squeeze(1)

param_grid = {
    "dropout": [0.0, 0.2, 0.3],
    "static_dim": [16, 32, 64], 
    "hidden_dim": [32, 64, 128],
    "batch_size": [32, 64],
    "num_layers": [1,2]
}

param_combinations = list(itertools.product(
    param_grid["dropout"],
    param_grid["static_dim"],
    param_grid["hidden_dim"],
    param_grid["batch_size"], 
    param_grid["num_layers"]
))

param_combinations = [
    (dropout, static_dim, hidden_dim, batch_size, num_layers)
    for (dropout, static_dim, hidden_dim, batch_size, num_layers) in param_combinations
    if not (dropout == 0.0 and num_layers > 1)
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"총 {len(param_combinations)}개의 하이퍼파라미터 조합을 탐색합니다. Device: {device}")

best_configs = {}
grid_results = []

for model_type in ["lstm", "gru"]:
    print(f"\n==============================")

    best_val_loss = float("inf")
    best_config = None

    for i, (dropout, static_dim, hidden_dim, batch_size, num_layers) in enumerate(param_combinations, 1):
        train_loader = DataLoader(LoadDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(LoadDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        input_static_dim = X_static_train.shape[1]
        if model_type == "lstm":
            model = LSTMForecast(input_static_dim, hidden_dim=hidden_dim, static_embed_dim=static_dim, 
                                 dropout=dropout, num_layers= num_layers)
        else:
            model = GRUForecast(input_static_dim, hidden_dim=hidden_dim, static_embed_dim=static_dim, 
                                dropout=dropout, num_layers= num_layers)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        n_epochs = 20

        print(f"\n---GridSearch {i}/{len(param_combinations)} 조합 시작: "
          f"dropout={dropout}, static_dim={static_dim}, hidden_dim={hidden_dim}, "
          f"batch_size={batch_size}, num_layers={num_layers}")
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for past, static, target in train_loader:
                past, static, target = past.to(device), static.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(past, static)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"   Epoch {epoch+1}/{n_epochs} 완료 | Train Loss: {avg_loss:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for past, static, target in val_loader:
                past, static, target = past.to(device), static.to(device), target.to(device)
                out = model(past, static)
                val_loss += criterion(out, target).item()
        val_loss /= len(val_loader)

        grid_results.append({
            "model": model_type, "dropout": dropout, "static_dim": static_dim,
            "hidden_dim": hidden_dim, "batch_size": batch_size, "num_layers":num_layers, "val_loss": val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = {
                "dropout": dropout, "static_dim": static_dim,
                "hidden_dim": hidden_dim, "batch_size": batch_size, "num_layers":num_layers, "val_loss": val_loss
            }

        print(f"[{model_type.upper()}] {i}/{len(param_combinations)} -> val_loss: {val_loss:.6f}")

    best_configs[model_type] = best_config
    print(f" {model_type.upper()} Best Config:", best_config)

pd.DataFrame(grid_results).to_csv(os.path.join(output_dir, "grid_search_results.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(best_configs).T.to_csv(os.path.join(output_dir, "best_configs.csv"), index=False, encoding="utf-8-sig")

results = {}

for model_type in ["lstm", "gru"]:
    cfg = best_configs.get(model_type)
    if cfg is None:
        print(f"{model_type}에 대한 best config가 없습니다.")
        continue

    print(f"\n[{model_type.upper()}] 최종 학습 시작 (Best Config): {cfg}")

    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(LoadDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(LoadDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(LoadDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_static_dim = X_static_train.shape[1]
    if model_type == "lstm":
        model = LSTMForecast(input_static_dim, hidden_dim=int(cfg["hidden_dim"]),
                             static_embed_dim=int(cfg["static_dim"]), dropout=float(cfg["dropout"]))
    else:
        model = GRUForecast(input_static_dim, hidden_dim=int(cfg["hidden_dim"]),
                            static_embed_dim=int(cfg["static_dim"]), dropout=float(cfg["dropout"]))
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100
    best_path = os.path.join(model_dir, f"{model_type}_best.pt")
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for past, static, target in train_loader:
            past, static, target = past.to(device), static.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(past, static)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for past, static, target in val_loader:
                past, static, target = past.to(device), static.to(device), target.to(device)
                val_loss += criterion(model(past, static), target).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        print(f"[{model_type.upper()}] Epoch {epoch+1}/{n_epochs} | TrainLoss: {train_loss/len(train_loader):.6f} | ValLoss: {val_loss:.6f}")

    # test
    model.load_state_dict(torch.load(best_path))
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for past, static, target in test_loader:
            past, static = past.to(device), static.to(device)
            out = model(past, static).cpu().numpy()
            preds_scaled.extend(out)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)

    y_true_inv = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_inv = scaler_y.inverse_transform(preds_scaled).flatten()

    eps = 1e-8
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2 = r2_score(y_true_inv, y_pred_inv)
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / (np.where(np.abs(y_true_inv) < eps, eps, y_true_inv)))) * 100

    results[model_type] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
    print(f"[{model_type.upper()}] Test Results -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")

    # 예측 결과 저장
    pred_df = pd.DataFrame({"datetime": time_test, "y_true": y_true_inv, "y_pred": y_pred_inv})
    pred_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------
    # 플롯들 저장 (30일 샘플 / 1년 전체 / 잔차 / 히스토그램 / ACF)
    # ----------------------------
    # 30일 샘플
    days_plot = 30
    n_points_30d = min(len(time_test), 24 * days_plot)
    plt.figure(figsize=(12,5))
    plt.plot(time_test[:n_points_30d], y_true_inv[:n_points_30d], label="True")
    plt.plot(time_test[:n_points_30d], y_pred_inv[:n_points_30d], linestyle="--", label="Pred")
    plt.title(f"{model_type.upper()} - True vs Pred (30 days)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "test_result_30days.png"))
    plt.close()

    # 1년 전체
    plt.figure(figsize=(15,6))
    plt.plot(time_test, y_true_inv, label="True", linewidth=1)
    plt.plot(time_test, y_pred_inv, linestyle="--", label="Pred", linewidth=1)
    plt.title(f"{model_type.upper()} - True vs Pred (Full Year)")
    plt.xlabel("DateTime"); plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "test_result_fullyear.png"))
    plt.close()

    # 잔차 및 히스토그램
    residuals = y_true_inv - y_pred_inv
    plt.figure(figsize=(12,4))
    plt.plot(time_test[:n_points_30d], residuals[:n_points_30d])
    plt.title(f"{model_type.upper()} - Residuals (30 days)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "residuals_30days.png"))
    plt.close()

    plt.figure(figsize=(17,4))
    plt.plot(time_test, residuals)
    plt.title(f"{model_type.upper()} - Residuals (1 year)")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "residuals_1year.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=50)
    plt.title(f"{model_type.upper()} - Residual Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "residual_hist.png"))
    plt.close()

    # ACF
    max_len_acf = min(len(residuals), 24*365)
    try:
        plt.figure(figsize=(13,6))
        plot_acf(residuals[:max_len_acf], lags=min(750, max_len_acf-1))
        plt.title(f"{model_type.upper()} - Residual Autocorrelation (up to 750 lags)")
        plt.xlabel("Lags (hours)"); plt.ylabel("Autocorrelation")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "residual_acf_1year.png"))
        plt.close()
    except Exception as e:
        print("ACF 계산 중 오류:", e)

    # ----------------------------
    # SHAP 분석
    # ----------------------------
    try:
        print("SHAP 분석 시작 (KernelExplainer")
        X_past_test_scaled = X_test[:, :24]
        X_static_test_scaled = X_test[:, 24:]
        X_full_test_scaled = X_test

        background_size = min(100, len(X_full_test_scaled))
        sample_size = min(200, len(X_full_test_scaled))

        if background_size < 5:
            print("테스트 데이터가 너무 작아 SHAP을 건너뜁니다.")
        else:
            background_idx = np.random.choice(len(X_full_test_scaled), background_size, replace=False)
            sample_idx = np.random.choice(len(X_full_test_scaled), sample_size, replace=False)

            X_background = X_full_test_scaled[background_idx]
            X_sample = X_full_test_scaled[sample_idx]

            def model_forward_numpy(X_np):
                past_np = torch.tensor(X_np[:, :24].reshape(-1,24,1), dtype=torch.float32).to(device)
                static_np = torch.tensor(X_np[:, 24:], dtype=torch.float32).to(device)
                model.eval()
                with torch.no_grad():
                    out = model(past_np, static_np).cpu().numpy().reshape(-1)
                return out

            explainer = shap.KernelExplainer(model_forward_numpy, X_background)
            shap_values = explainer.shap_values(X_sample, nsamples=200)

            feature_names = past_load_cols + static_cols
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}).sort_values(by="mean_abs_shap", ascending=False)
            shap_df.to_csv(os.path.join(model_dir, "shap_importance.csv"), index=False, encoding="utf-8-sig")

            plt.figure(figsize=(8,6))
            top10 = shap_df.head(10).iloc[::-1]
            plt.barh(top10['feature'], top10['mean_abs_shap'])
            plt.title(f"Top 10 Features by SHAP ({model_type.upper()})")
            plt.xlabel("Mean |SHAP value|")
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "shap_top10.png"))
            plt.close()
            print("SHAP 분석 완료")
    except Exception as e:
        print("SHAP 분석 중 오류 발생:", e)
    pd.DataFrame([{"model": model_type, **results[model_type]}]).to_csv(os.path.join(model_dir, "test_metrics.csv"), index=False, encoding="utf-8-sig")


print("\nFinal Result")
result_df = pd.DataFrame(results).T
print(result_df)
result_df.to_csv(os.path.join(output_dir, "final_results.csv"), encoding="utf-8-sig")
