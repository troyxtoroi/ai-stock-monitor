import json
import os
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd

TW_TZ = pytz.timezone('Asia/Taipei')


def now_tw():
    return datetime.now(TW_TZ)


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {filepath}")


def calc_indicators(hist):
    if hist.empty or len(hist) < 26:
        return {}
    close = hist['Close']

    ma5  = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    ema12  = close.ewm(span=12).mean()
    ema26  = close.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    return {
        "MA5":         round(float(ma5), 2),
        "MA10":        round(float(ma10), 2),
        "MA20":        round(float(ma20), 2),
        "RSI14":       round(float(rsi.iloc[-1]), 2),
        "MACD":        round(float(macd.iloc[-1]), 4),
        "MACD_signal": round(float(signal.iloc[-1]), 4),
        "MACD_hist":   round(float((macd - signal).iloc[-1]), 4),
    }


def fetch_quote(symbol):
    try:
        fi = yf.Ticker(symbol).fast_info
        price      = round(float(fi.last_price), 2)
        prev_close = round(float(fi.previous_close), 2)
        change     = round(price - prev_close, 2)
        change_pct = round(change / prev_close * 100, 2)
        return price, prev_close, change, change_pct
    except Exception:
        return None, None, None, None


def fetch_indices(watchlist):
    results = []
    for item in watchlist["indices"]:
        price, prev, chg, chg_pct = fetch_quote(item["symbol"])
        results.append({
            "symbol":     item["symbol"],
            "name":       item["name"],
            "price":      price,
            "prev_close": prev,
            "change":     chg,
            "change_pct": chg_pct,
            "updated":    now_tw().strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"  {item['name']}: {price} ({chg_pct}%)")
    return results


def fetch_stocks(watchlist):
    results = []
    for item in watchlist["stocks"]:
        price, prev, chg, chg_pct = fetch_quote(item["symbol"])
        try:
            hist = yf.Ticker(item["symbol"]).history(period="3mo")
            indicators = calc_indicators(hist)
            volume = int(hist["Volume"].iloc[-1]) if not hist.empty else None
        except Exception:
            indicators, volume = {}, None

        results.append({
            "symbol":     item["symbol"],
            "name":       item["name"],
            "price":      price,
            "prev_close": prev,
            "change":     chg,
            "change_pct": chg_pct,
            "volume":     volume,
            "indicators": indicators,
            "updated":    now_tw().strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"  {item['name']}: {price} ({chg_pct}%) RSI={indicators.get('RSI14','?')}")
    return results


def fetch_news(watchlist):
    all_news = []
    symbols = [s["symbol"] for s in watchlist["stocks"][:5]]
    seen = set()
    for symbol in symbols:
        try:
            items = yf.Ticker(symbol).news or []
            for n in items[:5]:
                title = n.get("title", "")
                if title in seen:
                    continue
                seen.add(title)
                ts = n.get("providerPublishTime", 0)
                all_news.append({
                    "title":    title,
                    "link":     n.get("link", ""),
                    "source":   n.get("publisher", ""),
                    "time":     datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "",
                    "related":  symbol,
                })
        except Exception as e:
            print(f"  news error ({symbol}): {e}")
    return all_news


def main():
    with open("config/watchlist.json", "r", encoding="utf-8") as f:
        watchlist = json.load(f)

    now = now_tw()
    date_str = now.strftime("%Y-%m-%d")
    print(f"\n=== 抓取時間: {now.strftime('%Y-%m-%d %H:%M:%S')} TST ===\n")

    print("[大盤指數]")
    indices = fetch_indices(watchlist)

    print("\n[個股行情 + 技術指標]")
    stocks = fetch_stocks(watchlist)

    print("\n[市場新聞]")
    news = fetch_news(watchlist)
    print(f"  共 {len(news)} 則新聞")

    save_json(indices, "data/index/latest.json")
    save_json(indices, f"data/index/{date_str}.json")
    save_json(stocks,  "data/stocks/latest.json")
    save_json(stocks,  f"data/stocks/{date_str}.json")
    save_json(news,    "data/news/latest.json")

    summary = {
        "updated":      now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone":     "Asia/Taipei",
        "indices":      indices,
        "stocks_count": len(stocks),
        "news_count":   len(news),
    }
    save_json(summary, "data/summary.json")
    print("\n完成！")


if __name__ == "__main__":
    main()
