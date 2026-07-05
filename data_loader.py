"""
data_loader.py - 한국석유공사 인천지역 경유 가격 CSV 로더
- 가로형(주유소 x 월별) 데이터를 월별 평균 시계열로 변환
- 이상치(정상 경유가 1000~3000원 밖) 자동 제거
"""
import os
import pandas as pd
import numpy as np

RAW_PATH = "incheon_diesel.csv"  # 한국석유공사_인천지역_경유_가격_현황

def load_prices():
    """반환: (labels, prices) — 월 라벨과 월평균 경유가격(원/L)"""
    if not os.path.exists(RAW_PATH):
        print(f"[경고] '{RAW_PATH}' 없음. 공공데이터포털에서 "
              f"'한국석유공사_인천지역 경유 가격 현황'을 받아 이 폴더에 저장하세요.")
        raise SystemExit(1)

    # 인코딩 자동 감지
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(RAW_PATH, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue

    month_cols = [c for c in df.columns if "년" in c]
    # 이상치 제거 후 월별 평균
    vals = df[month_cols].replace(0, pd.NA)
    clean = vals.where((vals > 1000) & (vals < 3000))
    avg = clean.mean().round(0)

    print(f"[실제 데이터] {RAW_PATH}: 주유소 {len(df)}개, {len(month_cols)}개월")
    print(f"  {month_cols[0]} {avg.iloc[0]:.0f}원 → {month_cols[-1]} {avg.iloc[-1]:.0f}원 "
          f"(+{(avg.iloc[-1]/avg.iloc[0]-1)*100:.1f}%)")
    return list(avg.index), avg.values

if __name__ == "__main__":
    labels, prices = load_prices()
    print("\n월별 평균 경유가격:")
    for l, p in zip(labels, prices):
        print(f"  {l}: {p:.0f}원")
