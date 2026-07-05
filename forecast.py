"""
forecast.py - 인천 경유 실데이터 기반 다음달 가격 예측
사용법: python forecast.py
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
from data_loader import load_prices

# 한글 폰트
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try:
        fm.fontManager.addfont(fp); plt.rcParams['font.family']='NanumGothic'
    except: pass
plt.rcParams["axes.unicode_minus"] = False

labels, prices = load_prices()
prices = np.array(prices); n = len(prices)

# 다음달 예측: 최근 3개월 선형 추세
x = np.arange(n).reshape(-1,1)
model = LinearRegression().fit(x[-3:], prices[-3:])
next_pred = model.predict([[n]])[0]

change = (next_pred/prices[-1]-1)*100
print(f"\n[예측] 다음달 경유가: {next_pred:.0f}원 ({change:+.1f}%)")
print(f"[급등] 최근 2개월: {prices[-3]:.0f} → {prices[-1]:.0f}원 (+{(prices[-1]/prices[-3]-1)*100:.1f}%)")

# 시각화
short = [l.replace('년','.').replace('월','') for l in labels]
plt.figure(figsize=(11,4.5))
plt.plot(range(n), prices, marker='o', color='#12130f', lw=2, label='실제 (한국석유공사)')
plt.plot([n-1,n], [prices[-1],next_pred], marker='o', color='#c1361c', lw=2, ls='--', label='다음달 예측')
plt.annotate('최근 2개월 급등', xy=(n-2, prices[-2]), xytext=(n-5, prices[-1]),
             fontsize=11, color='#c1361c', weight='bold',
             arrowprops=dict(arrowstyle='->', color='#c1361c'))
plt.xticks(list(range(n))+[n], short+['26.5'], rotation=45, fontsize=9)
plt.ylabel('경유 가격 (원/L)'); plt.title('인천지역 경유가격 추이와 다음달 예측', weight='bold')
plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
plt.savefig('forecast_result.png', dpi=120)
print("그래프 저장: forecast_result.png")
