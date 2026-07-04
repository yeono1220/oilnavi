"""
forecast.py - 실제 데이터 로더에 연결된 유가 예측 파이프라인
사용법: python forecast.py  (oil_raw.csv 있으면 실데이터, 없으면 시뮬레이션)
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from data_loader import load_prices

plt.rcParams["axes.unicode_minus"] = False
HORIZON = 14

# --- 1) 데이터 로드 (실제 or 시뮬레이션 자동) ---
df = load_prices("dubai")

# --- 2) 피처 엔지니어링 ---
for lag in [1,3,7,14,30]: df[f"lag_{lag}"] = df["price"].shift(lag)
for w in [7,14,30]:
    df[f"ma_{w}"] = df["price"].shift(1).rolling(w).mean()
    df[f"std_{w}"] = df["price"].shift(1).rolling(w).std()
df["mom_7"] = df["price"].shift(1) - df["price"].shift(8)
df["target"] = df["price"].shift(-HORIZON)
d = df.dropna().reset_index(drop=True)

feat = [c for c in d.columns if c.startswith(("lag_","ma_","std_","mom_"))]
X, y, now = d[feat].values, d["target"].values, d["price"].values
sp = int(len(d)*0.8)

# --- 3) 학습 + 평가 ---
m = LinearRegression().fit(X[:sp], y[:sp])
pred = m.predict(X[sp:])
yte, now_te = y[sp:], now[sp:]
mae = mean_absolute_error(yte, pred)
mape = mean_absolute_percentage_error(yte, pred)*100
da = ((yte>now_te)==(pred>now_te)).mean()*100
print(f"\n[예측 성능]  MAE ${mae:.2f} | MAPE {mape:.2f}% | 방향적중 {da:.1f}%")

# --- 4) 시각화 ---
dates_te = d["date"].iloc[sp:].values
plt.figure(figsize=(12,4.5))
plt.plot(dates_te, yte, color="#12130f", lw=1.5, label="Actual (14d later)")
plt.plot(dates_te, pred, color="#c1361c", lw=1.4, ls="--", label="Predicted")
plt.title("Oil Price Forecast (14-day ahead)", weight="bold")
plt.ylabel("Price ($/barrel)"); plt.legend(); plt.grid(alpha=.25)
plt.tight_layout(); plt.savefig("forecast_result.png", dpi=120)
print("그래프 저장: forecast_result.png")
