# 强势回踩选股大师

一个可以直接部署到 Streamlit Community Cloud 的日本股票筛选网站。

## 你的核心交易纪律已经写死在程序里

- 原本就强 + 回踩不破关键位 + 再次转强，才考虑买。
- 连续创新低的弱票重罚，不因“跌很多”自动抄底。
- 放量不涨也算弱，不把成交量本身当利好。
- 突然暴拉且远离短均线时增加追高惩罚。
- 新闻只作为催化确认，不能替代价格趋势。

## 股票池

共 71 只，无重复。

## 数据源

- 行情：Yahoo Finance / yfinance（免费，无 API key；可能延迟或缺失，不等于交易所级实时）
- 新闻：Google News RSS（免费，无 API key）

## 部署到 Streamlit Community Cloud

1. 新建 GitHub 仓库。
2. 上传本压缩包解压后的 `app.py`、`requirements.txt`、`README.md`。
3. 打开 https://share.streamlit.io/ ，选择仓库。
4. Main file path 选 `app.py`。
5. Deploy。

不需要填写 API Key。

## 本地启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 注意

免费公开行情不保证毫秒级实时，盘中实际下单前请用你的券商盘口确认。网站只做候选排序，不自动下单。
