import math
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="强势回踩选股大师", page_icon="📈", layout="wide")

JST = timezone(timedelta(hours=9))

STOCK_CODES = [
    "6203","485A","634A","6904","4073","6533","625A","5706","2980","5802","6920","7013","7012","8593","5016","6208","4274","4440","4052","604A","8848","6330","5801","8267","3103","3110","3003","200A","3350","3231","3101","5713","6701","4564","1615","4063","2502","7203","7261","3905","5074","4901","6526","8001","8306","2768","1570","6305","4062","285A","5803","6758","4755","9432","8136","4661","3038","282A","9984","7974","8035","6146","6857","6981","3778","3099","6976","6367","6506","7936","8058"
]

RULES = [
    "原本就强 + 回踩不破关键位 + 再次转强，才考虑买。",
    "连续创新低、收盘重心持续下移的弱票，直接重罚；不能因为“跌很多了”就自动抄底。",
    "有量不等于强：如果放量但价格不涨、冲高回落、收盘仍弱，视为派发/承接不足风险。",
    "类似 akippa：没有趋势、没有材料、没有惊喜，只靠突然暴拉，不追。",
    "关注新闻催化、最近走势、当前所处价位；新闻只作催化佐证，不替代价格确认。",
    "突然单日暴涨、明显远离短期均线时，即使评分高也增加追高惩罚。",
]

POSITIVE_KW = [
    "上方修正","増益","増収","最高益","過去最高","上方修正","受注","大型受注","採用","提携","業務提携","資本提携",
    "自社株買い","増配","復配","承認","認可","特許","新製品","新サービス","黒字転換","TOB","MBO","買収","株主還元",
    "好調","急拡大","契約","共同開発","AI","半導体","防衛","データセンター","電線","電力","インフラ"
]
NEGATIVE_KW = [
    "下方修正","減益","減収","赤字","赤字転落","業績悪化","不正","調査","行政処分","訴訟","事故","リコール","減配",
    "希薄化","増資","MSワラント","新株予約権","公募増資","売出し","破産","民事再生","債務超過"
]


def ticker(code: str) -> str:
    return f"{code}.T"


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def pct(a, b):
    if b is None or not math.isfinite(b) or b == 0 or a is None or not math.isfinite(a):
        return np.nan
    return (a / b - 1) * 100.0


@st.cache_data(ttl=120, show_spinner=False)
def download_daily(codes_tuple):
    symbols = [ticker(c) for c in codes_tuple]
    data = yf.download(
        tickers=symbols,
        period="4mo",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=20,
    )
    return data


def extract_one_daily(batch: pd.DataFrame, code: str) -> pd.DataFrame:
    sym = ticker(code)
    try:
        if isinstance(batch.columns, pd.MultiIndex):
            if sym in batch.columns.get_level_values(0):
                df = batch[sym].copy()
            elif sym in batch.columns.get_level_values(1):
                df = batch.xs(sym, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            df = batch.copy()
        df = df.dropna(how="all")
        # Normalize possible lowercase/mixed labels.
        df.columns = [str(c).title() for c in df.columns]
        need = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        return df[need].copy() if need else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def calc_rsi(close: pd.Series, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def analyze_daily(df: pd.DataFrame, code: str):
    if df is None or df.empty or "Close" not in df.columns or len(df.dropna(subset=["Close"])) < 22:
        return None
    d = df.dropna(subset=["Close"]).copy()
    for col in ["Open","High","Low","Volume"]:
        if col not in d.columns:
            d[col] = np.nan
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    vol = d["Volume"].astype(float)

    c = safe_float(close.iloc[-1])
    prev = safe_float(close.iloc[-2])
    ma5 = safe_float(close.rolling(5).mean().iloc[-1])
    ma20 = safe_float(close.rolling(20).mean().iloc[-1])
    ma60 = safe_float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else np.nan
    h20 = safe_float(high.tail(20).max())
    l20 = safe_float(low.tail(20).min())
    h60 = safe_float(high.tail(min(60, len(high))).max())
    l60 = safe_float(low.tail(min(60, len(low))).min())
    r1 = pct(c, prev)
    r5 = pct(c, safe_float(close.iloc[-6])) if len(close) >= 6 else np.nan
    r20 = pct(c, safe_float(close.iloc[-21])) if len(close) >= 21 else np.nan
    r60 = pct(c, safe_float(close.iloc[-61])) if len(close) >= 61 else np.nan
    vol5 = safe_float(vol.tail(5).mean())
    vol20 = safe_float(vol.tail(20).mean())
    volume_ratio = vol5 / vol20 if math.isfinite(vol5) and math.isfinite(vol20) and vol20 > 0 else np.nan
    last_vol_ratio = safe_float(vol.iloc[-1]) / vol20 if math.isfinite(vol20) and vol20 > 0 else np.nan
    rsi = safe_float(calc_rsi(close).iloc[-1])

    # Trend quality: rising moving averages and positive medium-term momentum.
    ma20_prev5 = safe_float(close.rolling(20).mean().iloc[-6]) if len(close) >= 26 else np.nan
    strong_score = 0
    if math.isfinite(r20): strong_score += np.clip(r20 / 12 * 18, -18, 18)
    if c > ma20: strong_score += 10
    if math.isfinite(ma5) and ma5 > ma20: strong_score += 8
    if math.isfinite(ma20_prev5) and ma20 > ma20_prev5: strong_score += 8
    if math.isfinite(r60): strong_score += np.clip(r60 / 25 * 8, -8, 8)

    # Key support is the highest of useful nearby supports, but not above price.
    supports = [x for x in [ma20, safe_float(low.tail(10).min()), safe_float(close.tail(10).min())] if math.isfinite(x) and x <= c]
    support = max(supports) if supports else l20
    support_gap = pct(c, support) if math.isfinite(support) else np.nan
    from_h20 = pct(c, h20)  # negative below high
    range_pos = (c - l20) / (h20 - l20) * 100 if math.isfinite(h20) and math.isfinite(l20) and h20 > l20 else np.nan

    # Pullback-hold: near MA/support, but not breaking it; reward healthy retreat from high.
    pullback_score = 0
    if math.isfinite(support_gap):
        if 0 <= support_gap <= 4: pullback_score += 16
        elif 4 < support_gap <= 8: pullback_score += 9
        elif support_gap < 0: pullback_score -= 18
    if math.isfinite(from_h20):
        draw = -from_h20
        if 2 <= draw <= 10: pullback_score += 12
        elif draw < 1: pullback_score += 2
        elif draw > 18: pullback_score -= 10

    # Re-strengthening confirmation.
    turn_score = 0
    prev_high = safe_float(high.iloc[-2])
    if c > prev_high: turn_score += 12
    if math.isfinite(r1) and r1 > 0: turn_score += min(r1 * 2.2, 9)
    if len(close) >= 3 and c > safe_float(close.iloc[-2]) > safe_float(close.iloc[-3]): turn_score += 7
    if math.isfinite(last_vol_ratio) and last_vol_ratio >= 1.15 and math.isfinite(r1) and r1 > 0: turn_score += min((last_vol_ratio - 1) * 8, 6)

    # Weakness / 'akippa-style' penalties.
    penalties = 0
    flags = []
    recent_lows = low.tail(8).dropna().values
    lower_low_count = int(sum(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows))))
    is_new20_low = math.isfinite(l20) and safe_float(low.iloc[-1]) <= l20 * 1.002
    if is_new20_low:
        penalties += 25; flags.append("接近/刷新20日低点")
    if lower_low_count >= 5:
        penalties += 15; flags.append("近期低点持续下移")
    if math.isfinite(r5) and r5 < -5:
        penalties += min(abs(r5) * 1.3, 16); flags.append("5日趋势偏弱")
    if math.isfinite(r20) and r20 < -10:
        penalties += min(abs(r20) * 0.8, 16); flags.append("20日趋势偏弱")
    if math.isfinite(last_vol_ratio) and last_vol_ratio >= 1.5 and ((math.isfinite(r1) and r1 <= 0.5) or c < ma5):
        penalties += 12; flags.append("放量但价格没有同步走强")
    if math.isfinite(r1) and r1 >= 7:
        dist_ma5 = pct(c, ma5)
        if math.isfinite(dist_ma5) and dist_ma5 >= 5:
            penalties += 18; flags.append("突然暴拉且远离MA5，追高风险")

    technical = strong_score + pullback_score + turn_score - penalties
    technical = float(np.clip(technical, -50, 100))

    return {
        "代码": code,
        "现价": c,
        "日涨跌%": r1,
        "5日%": r5,
        "20日%": r20,
        "60日%": r60,
        "MA5": ma5,
        "MA20": ma20,
        "20日高": h20,
        "20日低": l20,
        "距20日高%": from_h20,
        "区间位置%": range_pos,
        "量比20日": last_vol_ratio,
        "RSI14": rsi,
        "关键支撑": support,
        "距支撑%": support_gap,
        "强势分": float(strong_score),
        "回踩分": float(pullback_score),
        "转强分": float(turn_score),
        "弱势惩罚": float(penalties),
        "技术总分": technical,
        "风险标签": "；".join(flags) if flags else "无明显弱势惩罚",
        "_df": d,
    }


def fetch_news(code: str, max_items=6):
    # No API key. Public RSS; availability depends on network / Google response.
    query = urllib.parse.quote(f"{code} 株 OR {code} 決算 OR {code} 提携 OR {code} 受注")
    url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            out.append({"title": title, "link": link, "pubDate": pub})
        return out
    except Exception:
        return []


def score_news(items):
    if not items:
        return 0, "未抓到近期公开新闻"
    score = 0
    pos_hits, neg_hits = [], []
    for it in items:
        t = it["title"]
        for kw in POSITIVE_KW:
            if kw in t:
                score += 2.5
                pos_hits.append(kw)
        for kw in NEGATIVE_KW:
            if kw in t:
                score -= 4
                neg_hits.append(kw)
    score = float(np.clip(score, -15, 15))
    if pos_hits and not neg_hits:
        label = "有潜在正面催化词：" + "、".join(sorted(set(pos_hits))[:5])
    elif neg_hits:
        label = "存在风险新闻词：" + "、".join(sorted(set(neg_hits))[:5])
    else:
        label = "有新闻，但未识别到明确催化/风险关键词"
    return score, label


def classify(row):
    score = row["综合分"]
    risk = row["风险标签"]
    # Hard gates matter more than raw score.
    if "刷新20日低点" in risk or (math.isfinite(row["20日%"]) and row["20日%"] < -12):
        return "回避"
    if score >= 55 and row["强势分"] >= 12 and row["回踩分"] >= 8 and row["转强分"] >= 6:
        return "优先观察 / 可等确认"
    if score >= 38 and row["强势分"] >= 8:
        return "观察"
    if score >= 22:
        return "中性"
    return "回避"


def add_news_to_candidates(df_rank: pd.DataFrame, n=12):
    records = df_rank.head(n).to_dict("records")
    news_map = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_news, r["代码"]): r["代码"] for r in records}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                items = fut.result()
            except Exception:
                items = []
            ns, label = score_news(items)
            news_map[code] = (ns, label, items)
    return news_map


def format_num(v, digits=1):
    return "—" if v is None or not math.isfinite(v) else f"{v:.{digits}f}"


def scan_all(codes):
    batch = download_daily(tuple(codes))
    rows = []
    raw = {}
    for c in codes:
        d = extract_one_daily(batch, c)
        a = analyze_daily(d, c)
        if a:
            raw[c] = a.pop("_df")
            rows.append(a)
    if not rows:
        return pd.DataFrame(), {}, {}
    base = pd.DataFrame(rows)
    base["新闻分"] = 0.0
    base["新闻判断"] = "未精查"
    base["综合分"] = base["技术总分"]
    base = base.sort_values("综合分", ascending=False).reset_index(drop=True)

    news_map = add_news_to_candidates(base, n=min(15, len(base)))
    for idx, row in base.iterrows():
        c = row["代码"]
        if c in news_map:
            ns, label, _items = news_map[c]
            base.at[idx, "新闻分"] = ns
            base.at[idx, "新闻判断"] = label
            base.at[idx, "综合分"] = float(np.clip(base.at[idx, "技术总分"] + ns, -50, 100))
    base["结论"] = base.apply(classify, axis=1)
    base = base.sort_values(["综合分", "20日%"], ascending=[False, False]).reset_index(drop=True)
    return base, raw, news_map


def chart_for(code, raw_df, support=None):
    d = raw_df.tail(60).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name=code))
    ma5 = d["Close"].rolling(5).mean()
    ma20 = d["Close"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=d.index, y=ma5, mode="lines", name="MA5"))
    fig.add_trace(go.Scatter(x=d.index, y=ma20, mode="lines", name="MA20"))
    if support is not None and math.isfinite(support):
        fig.add_hline(y=support, line_dash="dash", annotation_text="关键支撑")
    fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=35,b=10))
    return fig


# ---------- UI ----------
st.title("📈 强势回踩选股大师")
st.caption("规则引擎版 · 股票池固定 71 只 · 免费行情源 Yahoo Finance/yfinance（可能延迟，不保证交易所级实时）")

with st.expander("先看核心纪律（网站会强制执行）", expanded=True):
    for x in RULES:
        st.markdown(f"- **{x}**")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.metric("股票池", f"{len(STOCK_CODES)} 只")
with colB:
    st.metric("重复代码", f"{len(STOCK_CODES)-len(set(STOCK_CODES))} 个")
with colC:
    st.metric("数据刷新", datetime.now(JST).strftime("%Y-%m-%d %H:%M JST"))

st.warning("这不是自动下单系统。免费公开行情可能延迟或缺失；推荐结果是筛选/排序，不是收益保证。尤其盘中快速波动时，请用券商盘口确认价格后再行动。")

if "scan" not in st.session_state:
    st.session_state.scan = None
if "raw" not in st.session_state:
    st.session_state.raw = None
if "news" not in st.session_state:
    st.session_state.news = None

left, right = st.columns([1, 2])
with left:
    if st.button("🚀 扫描 71 只股票", type="primary", use_container_width=True):
        with st.spinner("正在拉取行情、计算趋势，并对前排候选精查新闻…"):
            rank, raw, news = scan_all(STOCK_CODES)
            st.session_state.scan = rank
            st.session_state.raw = raw
            st.session_state.news = news
with right:
    st.caption("扫描逻辑：先看趋势和价位 → 剔除连续创新低/放量不涨 → 判断回踩是否守住 → 判断是否重新转强 → 最后给前排候选叠加近期新闻催化。")

rank = st.session_state.scan
raw_map = st.session_state.raw or {}
news_map = st.session_state.news or {}

if rank is not None and not rank.empty:
    st.subheader("今天优先看谁")
    top = rank.iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("第一名", str(top["代码"]))
    c2.metric("综合分", format_num(top["综合分"],1))
    c3.metric("20日涨跌", f"{format_num(top['20日%'],1)}%")
    c4.metric("结论", top["结论"])
    st.info(f"第一名 {top['代码']}：{top['风险标签']}；{top['新闻判断']}。注意：第一名也必须等实际盘口确认，不等于无脑买。")

    st.markdown("#### 排名表")
    show_cols = ["代码","结论","综合分","现价","日涨跌%","5日%","20日%","距20日高%","距支撑%","量比20日","强势分","回踩分","转强分","弱势惩罚","新闻判断","风险标签"]
    display_df = rank[show_cols].copy()
    num_cols = [c for c in show_cols if c not in ["代码","结论","新闻判断","风险标签"]]
    display_df[num_cols] = display_df[num_cols].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=620)

    st.divider()
    st.subheader("问大师：现在买哪个？")
    q = st.text_input("可以直接问：‘现在买哪个？’、‘6203 和 5801 哪个更好？’、‘9984 还能追吗？’", placeholder="输入问题")
    if q:
        codes_in_q = [c for c in STOCK_CODES if re.search(rf"(?<![0-9A-Z]){re.escape(c)}(?![0-9A-Z])", q.upper())]
        scope = rank[rank["代码"].isin(codes_in_q)] if codes_in_q else rank.head(10)
        if scope.empty:
            st.error("没识别到股票池里的代码。直接输入代码最稳，例如：6203 和 5801 哪个更好？")
        else:
            best = scope.sort_values("综合分", ascending=False).iloc[0]
            if "追" in q and math.isfinite(best["日涨跌%"] or np.nan) and best["日涨跌%"] >= 6:
                st.error(f"{best['代码']} 今天已经明显拉升。按你的纪律：**不追突然暴拉**。等回踩守住关键位、再度转强再看。")
            else:
                st.success(f"当前规则下更优的是 **{best['代码']}**（综合分 {best['综合分']:.1f}，结论：{best['结论']}）。")
                st.write(
                    f"理由：20日 {format_num(best['20日%'])}%｜距20日高 {format_num(best['距20日高%'])}%｜距支撑 {format_num(best['距支撑%'])}%｜"
                    f"强势 {format_num(best['强势分'])} / 回踩 {format_num(best['回踩分'])} / 再转强 {format_num(best['转强分'])}｜弱势惩罚 {format_num(best['弱势惩罚'])}。"
                )
                st.write(f"风险：{best['风险标签']}。新闻：{best['新闻判断']}。")

    st.divider()
    st.subheader("单票诊断")
    selected = st.selectbox("股票代码", rank["代码"].tolist())
    row = rank[rank["代码"] == selected].iloc[0]
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("现价", format_num(row["现价"],1))
    m2.metric("日涨跌", f"{format_num(row['日涨跌%'])}%")
    m3.metric("20日", f"{format_num(row['20日%'])}%")
    m4.metric("距支撑", f"{format_num(row['距支撑%'])}%")
    m5.metric("综合分", format_num(row["综合分"]))
    st.write(f"**结论：{row['结论']}**｜{row['风险标签']}｜{row['新闻判断']}")
    if selected in raw_map:
        st.plotly_chart(chart_for(selected, raw_map[selected], row["关键支撑"]), use_container_width=True)
    if selected in news_map:
        items = news_map[selected][2]
    else:
        items = fetch_news(selected)
    if items:
        st.markdown("**近期公开新闻**")
        for it in items[:6]:
            title = it['title'].replace('[','').replace(']','')
            st.markdown(f"- [{title}]({it['link']})")
    else:
        st.caption("暂未抓到公开新闻；不要把‘没抓到’理解成‘公司没有新闻’。")
else:
    st.info("点击上面的“扫描 71 只股票”开始。首次加载可能稍慢。")

with st.expander("股票池（71只）"):
    st.code("\n".join(STOCK_CODES), language="text")

with st.expander("评分怎么判"):
    st.markdown("""
**1. 原本就强**：20日/60日动量、价格是否站上 MA20、MA5 是否高于 MA20、MA20 是否抬升。  
**2. 回踩不破**：价格靠近 MA20/近10日支撑，但没有有效跌破；从20日高点适度回撤比高位硬追更好。  
**3. 再次转强**：收盘重新越过前一日高点、连续回升、上涨同时伴随合理放量。  
**4. akippa式弱票惩罚**：接近/刷新20日低点、近期低点连续下移、5日/20日持续走弱、放量不涨。  
**5. 追高惩罚**：单日突然大涨且明显远离 MA5，直接扣分。  
**6. 新闻**：只对前排候选精查公开新闻，识别上方修正、受注、提携、自社株买、增配等潜在催化，以及下方修正、增资、MS warrant 等风险词。新闻不会覆盖掉技术面硬伤。
""")

st.caption("数据说明：Yahoo Finance/yfinance 为免费公开数据入口，不是东京证券交易所官方低延迟行情。新闻来自 Google News RSS。")
