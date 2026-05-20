import json
import os
from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import anthropic

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
    for cat in watchlist["categories"]:
        for item in cat["stocks"]:
            price, prev, chg, chg_pct = fetch_quote(item["symbol"])
            try:
                hist = yf.Ticker(item["symbol"]).history(period="3mo")
                indicators = calc_indicators(hist)
                volume = int(hist["Volume"].iloc[-1]) if not hist.empty else None
            except Exception:
                indicators, volume = {}, None

            results.append({
                "symbol":      item["symbol"],
                "name":        item["name"],
                "category_id": cat["id"],
                "category":    cat["name"],
                "price":       price,
                "prev_close":  prev,
                "change":      chg,
                "change_pct":  chg_pct,
                "volume":      volume,
                "indicators":  indicators,
                "updated":     now_tw().strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"  [{cat['name']}] {item['name']}: {price} ({chg_pct}%) RSI={indicators.get('RSI14','?')}")
    return results


def fetch_news(watchlist):
    all_news = []
    all_stocks = [s for cat in watchlist["categories"] for s in cat["stocks"]]
    symbols = [s["symbol"] for s in all_stocks[:5]]
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


def ai_analyze(stocks, indices):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  跳過 AI 分析（未設定 ANTHROPIC_API_KEY）")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    index_summary = " | ".join(
        f"{i['name']} {i['price']}（{'+' if (i['change_pct'] or 0) > 0 else ''}{i['change_pct']}%）"
        for i in indices
    )

    stocks_text = "\n".join(
        f"{s['name']}({s['symbol']}): 股價{s['price']} 漲跌{s['change_pct']}% "
        f"MA5={s['indicators'].get('MA5','?')} MA10={s['indicators'].get('MA10','?')} MA20={s['indicators'].get('MA20','?')} "
        f"RSI={s['indicators'].get('RSI14','?')} "
        f"MACD={s['indicators'].get('MACD','?')} MACD_signal={s['indicators'].get('MACD_signal','?')} MACD_hist={s['indicators'].get('MACD_hist','?')}"
        for s in stocks
    )

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": (
                    "你是專業台股技術分析師，擅長根據技術指標給出明確的買賣操作建議。\n"
                    "分析規則：\n"
                    "1. 趨勢判斷：股價>MA5>MA10>MA20 為強勢上升；股價<MA5<MA10<MA20 為強勢下降；其餘為盤整\n"
                    "2. RSI>70 超買（偏賣）；RSI<30 超賣（偏買）；50~70 偏多；30~50 偏空\n"
                    "3. MACD_hist>0 且擴大 為買入訊號；MACD_hist<0 且擴大 為賣出訊號\n"
                    "4. 綜合三項指標給出明確操作建議\n"
                    "輸出嚴格遵守JSON格式，不加任何說明文字。"
                ),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": f"""大盤狀況：{index_summary}

各股技術指標：
{stocks_text}

請針對每支股票輸出明確操作建議，JSON陣列格式：
[{{
  "symbol": "股票代號",
  "name": "股票名稱",
  "trend": "上升|下降|盤整",
  "signal": "買入|賣出|觀望",
  "confidence": "高|中|低",
  "reason": "具體說明為何買入或賣出（60字內，要有數據依據）",
  "risk": "主要風險提示（30字內）"
}}]"""}],
        )
        results = json.loads(resp.content[0].text)
        ts = now_tw().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            r["updated"] = ts
        print(f"  AI 分析完成，共 {len(results)} 支")
        return results
    except Exception as e:
        print(f"  AI 分析失敗：{e}")
        return []


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

    print("\n[AI 分析訊號]")
    ai_signals = ai_analyze(stocks, indices)
    if ai_signals:
        save_json(ai_signals, "data/ai_signals.json")

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
