import json
import os
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd

TW_TZ = pytz.timezone('Asia/Taipei')


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))


def to_series(index, values):
    result = []
    for date, val in zip(index, values):
        if not pd.isna(val):
            result.append({'time': date.strftime('%Y-%m-%d'), 'value': round(float(val), 2)})
    return result


def process_hist(hist):
    if hist is None or hist.empty or len(hist) < 20:
        return None

    close  = hist['Close']
    high   = hist['High']
    low    = hist['Low']
    open_  = hist['Open']

    # K 線
    candles = []
    for date, row in hist.iterrows():
        candles.append({
            'time':  date.strftime('%Y-%m-%d'),
            'open':  round(float(row['Open']),  2),
            'high':  round(float(row['High']),  2),
            'low':   round(float(row['Low']),   2),
            'close': round(float(row['Close']), 2),
        })

    # 成交量
    volume = []
    for date, row in hist.iterrows():
        vol = int(row['Volume']) if not pd.isna(row['Volume']) else 0
        color = '#3fb950' if row['Close'] >= row['Open'] else '#f85149'
        volume.append({'time': date.strftime('%Y-%m-%d'), 'value': vol, 'color': color})

    # 均線
    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # 布林通道（20, 2σ）
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # KD 隨機指標（9,3,3）
    low_min  = low.rolling(9).min()
    high_max = high.rolling(9).max()
    denom    = high_max - low_min
    rsv = pd.Series(
        [(c - lo) / de * 100 if de and de != 0 else 50
         for c, lo, de in zip(close, low_min, denom)],
        index=close.index
    )
    kd_k = rsv.ewm(com=2, adjust=False).mean()
    kd_d = kd_k.ewm(com=2, adjust=False).mean()

    # 黃金交叉 / 死亡交叉（MA5 穿越 MA20）
    crosses = []
    ma5v  = ma5.values
    ma20v = ma20.values
    dates = list(hist.index)
    for i in range(1, len(dates)):
        p5, p20 = ma5v[i-1], ma20v[i-1]
        c5, c20 = ma5v[i],   ma20v[i]
        if any(pd.isna(x) for x in [p5, p20, c5, c20]):
            continue
        date_str = dates[i].strftime('%Y-%m-%d')
        price    = round(float(close.iloc[i]), 2)
        if p5 < p20 and c5 >= c20:
            crosses.append({'time': date_str, 'type': 'golden', 'price': price})
        elif p5 > p20 and c5 <= c20:
            crosses.append({'time': date_str, 'type': 'death',  'price': price})

    return {
        'candles':  candles,
        'volume':   volume,
        'ma5':      to_series(hist.index, ma5),
        'ma10':     to_series(hist.index, ma10),
        'ma20':     to_series(hist.index, ma20),
        'ma60':     to_series(hist.index, ma60),
        'bb_upper': to_series(hist.index, bb_upper),
        'bb_mid':   to_series(hist.index, bb_mid),
        'bb_lower': to_series(hist.index, bb_lower),
        'kd_k':     to_series(hist.index, kd_k),
        'kd_d':     to_series(hist.index, kd_d),
        'crosses':  crosses[-30:],
    }


def main():
    with open('config/watchlist.json', 'r', encoding='utf-8') as f:
        watchlist = json.load(f)

    all_stocks = [s for cat in watchlist['categories'] for s in cat['stocks']]
    print(f'\n=== 更新K線圖資料（共 {len(all_stocks)} 檔）===\n')

    for stock in all_stocks:
        symbol, name = stock['symbol'], stock['name']
        try:
            ticker = yf.Ticker(symbol)

            daily  = process_hist(ticker.history(period='6mo', interval='1d'))
            weekly = process_hist(ticker.history(period='2y',  interval='1wk'))

            if daily:
                save_json(daily,  f'data/charts/{symbol}_daily.json')
            if weekly:
                save_json(weekly, f'data/charts/{symbol}_weekly.json')

            d = len(daily['candles'])  if daily  else 0
            w = len(weekly['candles']) if weekly else 0
            x = len(daily['crosses'])  if daily  else 0
            print(f'  {name}({symbol}): 日線{d}筆 週線{w}筆 交叉{x}次')
        except Exception as e:
            print(f'  {name}({symbol}) 錯誤: {e}')

    print('\nK線圖資料更新完成！')


if __name__ == '__main__':
    main()
