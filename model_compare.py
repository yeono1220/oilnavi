"""
model_compare.py - 선형/부스팅/LSTM 3종 비교 실험
핵심: 유가 시계열에서 '딥러닝이 항상 이기지 않는다'를 데이터로 검증
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from data_loader import load_prices

np.random.seed(42); torch.manual_seed(42)
plt.rcParams["axes.unicode_minus"] = False
HOR = 14

df = load_prices("dubai")
price = df["price"].values

# 전통 모델용 피처
dd = pd.DataFrame({"price": price})
for l in [1,3,7,14,30]: dd[f"lag{l}"] = dd.price.shift(l)
for w in [7,14,30]:
    dd[f"ma{w}"] = dd.price.shift(1).rolling(w).mean()
    dd[f"sd{w}"] = dd.price.shift(1).rolling(w).std()
dd["mom7"] = dd.price.shift(1) - dd.price.shift(8)
dd["y"] = dd.price.shift(-HOR)
dd = dd.dropna().reset_index(drop=True)
feat = [c for c in dd.columns if c.startswith(("lag","ma","sd","mom"))]
X, y, now = dd[feat].values, dd.y.values, dd.price.values
sp = int(len(dd)*0.8)
res = {}

def report(name, yte, pred, now_te):
    mae = mean_absolute_error(yte, pred)
    mape = mean_absolute_percentage_error(yte, pred)*100
    da = ((yte>now_te)==(pred>now_te)).mean()*100
    print(f"{name:22s} | MAE ${mae:5.2f} | MAPE {mape:5.2f}% | dir {da:5.1f}%")
    return mae, mape, da

m = LinearRegression().fit(X[:sp], y[:sp])
res["Linear"] = report("Linear (baseline)", y[sp:], m.predict(X[sp:]), now[sp:])
m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42).fit(X[:sp], y[:sp])
res["GBM"] = report("GradientBoosting", y[sp:], m.predict(X[sp:]), now[sp:])

# LSTM
SEQ = 30
xs, ys, nows = [], [], []
for i in range(SEQ, len(price)-HOR):
    xs.append(price[i-SEQ:i]); ys.append(price[i+HOR]); nows.append(price[i])
xs = np.array(xs)[:,:,None]; ys = np.array(ys); nows = np.array(nows)
sc = StandardScaler(); sc.fit(xs.reshape(-1,1)[:sp*SEQ])
xs_s = sc.transform(xs.reshape(-1,1)).reshape(xs.shape)
ymean, ystd = ys[:sp].mean(), ys[:sp].std(); ys_s = (ys-ymean)/ystd
spx = int(len(xs)*0.8)
Xtr = torch.tensor(xs_s[:spx], dtype=torch.float32); ytr = torch.tensor(ys_s[:spx], dtype=torch.float32)
Xte = torch.tensor(xs_s[spx:], dtype=torch.float32)

class LSTM(nn.Module):
    def __init__(s):
        super().__init__(); s.l = nn.LSTM(1,32,batch_first=True); s.f = nn.Linear(32,1)
    def forward(s,x): o,_ = s.l(x); return s.f(o[:,-1,:]).squeeze(-1)

net = LSTM(); opt = torch.optim.Adam(net.parameters(), lr=0.01); lossf = nn.MSELoss()
for ep in range(60):
    net.train(); opt.zero_grad(); loss = lossf(net(Xtr), ytr); loss.backward(); opt.step()
net.eval()
with torch.no_grad(): pred = net(Xte).numpy()*ystd + ymean
res["LSTM"] = report("LSTM (deep)", ys[spx:], pred, nows[spx:])

best = min(res, key=lambda k: res[k][1])
print(f"\n결론: MAPE 최고 = {best}  → 단순 모델이 딥러닝을 이기는 경우가 흔함(과적합).")

# 그래프
fig, ax = plt.subplots(1,2, figsize=(12,4.2))
names = list(res.keys()); mapes = [res[k][1] for k in names]; das = [res[k][2] for k in names]
c = ["#457b9d","#e9c46a","#e63946"]
ax[0].bar(names, mapes, color=c); ax[0].set_title("MAPE (lower=better)", weight="bold"); ax[0].set_ylabel("%")
for i,v in enumerate(mapes): ax[0].text(i,v+.05,f"{v:.2f}",ha="center",weight="bold")
ax[1].bar(names, das, color=c); ax[1].axhline(50,ls="--",color="gray",label="random")
ax[1].set_title("Direction Accuracy (higher=better)", weight="bold"); ax[1].set_ylabel("%"); ax[1].legend()
for i,v in enumerate(das): ax[1].text(i,v+.5,f"{v:.1f}",ha="center",weight="bold")
plt.tight_layout(); plt.savefig("model_compare.png", dpi=120)
print("그래프 저장: model_compare.png")
