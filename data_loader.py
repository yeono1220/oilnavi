"""
data_loader.py
- 한국석유공사 국제유가 CSV를 읽어 예측 파이프라인에 연결
- CSV가 있으면 실제 데이터, 없으면 시뮬레이션으로 자동 폴백
- 공공데이터 CSV는 인코딩(cp949/euc-kr)과 컬럼명이 제각각이라 자동 감지 처리

[실제 데이터 받는 법]
1. https://www.data.go.kr 접속 → "한국석유공사 국제유가" 검색
   또는 오피넷 https://www.opinet.co.kr → 국제유가 → 유종별 엑셀 다운로드
2. 받은 CSV/XLSX를 이 폴더에 'oil_raw.csv' 로 저장
3. python data_loader.py 실행
"""
import os
import numpy as np
import pandas as pd

RAW_PATH = "oil_raw.csv"   # 여기에 석유공사 파일을 두면 됨


def _read_any(path):
    """공공데이터 CSV/XLSX를 인코딩 자동 감지로 읽기."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("CSV 인코딩을 읽지 못했습니다. 파일을 확인하세요.")


def _find_col(cols, keywords):
    """컬럼명에서 키워드가 포함된 첫 컬럼 찾기 (날짜/가격 자동 매칭)."""
    for c in cols:
        cl = str(c).replace(" ", "").lower()
        if any(k in cl for k in keywords):
            return c
    return None


def load_prices(kind="dubai"):
    """
    반환: DataFrame(date, price)  — 날짜 오름차순 정렬
    kind: 'dubai' | 'wti' | 'brent'  (컬럼 자동 탐색용 힌트)
    """
    if os.path.exists(RAW_PATH):
        df = _read_any(RAW_PATH)
        cols = list(df.columns)

        # 날짜 컬럼 자동 탐색
        date_col = _find_col(cols, ["date", "일자", "날짜", "기준일"])
        # 가격 컬럼: 유종 키워드 우선, 없으면 '가격/price/종가'
        hint = {"dubai": ["dubai", "두바이"], "wti": ["wti"], "brent": ["brent", "브렌트"]}[kind]
        price_col = _find_col(cols, hint) or _find_col(cols, ["price", "가격", "종가", "$", "달러"])

        if date_col is None or price_col is None:
            print(f"[경고] 컬럼 자동인식 실패. 발견된 컬럼: {cols}")
            print("      코드 상단에서 date_col/price_col을 직접 지정하세요.")
            raise SystemExit(1)

        out = df[[date_col, price_col]].copy()
        out.columns = ["date", "price"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["price"] = pd.to_numeric(
            out["price"].astype(str).str.replace(",", "").str.replace("$", ""),
            errors="coerce"
        )
        out = out.dropna().sort_values("date").reset_index(drop=True)
        print(f"[실제 데이터] {RAW_PATH} 로드: {len(out)}행, "
              f"{out.date.min().date()} ~ {out.date.max().date()}, "
              f"${out.price.min():.1f}~${out.price.max():.1f}")
        return out

    # ---- 폴백: 시뮬레이션 (실제 데이터 없을 때) ----
    print(f"[시뮬레이션] '{RAW_PATH}'가 없어 예시 데이터로 실행합니다. "
          f"실제 데이터를 넣으려면 파일을 이 폴더에 저장하세요.")
    np.random.seed(42)
    n = 1500
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    sh = np.random.normal(0, 1.1, n)
    price = np.clip(68 + np.cumsum(sh) * 0.5 + 3 * np.sin(np.arange(n) * 2 * np.pi / 365), 20, None)
    return pd.DataFrame({"date": dates, "price": price})


if __name__ == "__main__":
    df = load_prices("dubai")
    print(df.tail())
    print("\n연결 성공. 이제 oil_forecast.py / oil_deep.py 가 이 로더를 사용합니다.")
