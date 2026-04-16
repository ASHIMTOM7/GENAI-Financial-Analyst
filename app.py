import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from transformers import pipeline

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gen AI Financial Analyst", layout="wide")

# ------------------------------
# LOAD MODEL (SAFE)
# ------------------------------
@st.cache_resource
def load_model():
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    except:
        return None

sentiment_pipeline = load_model()

# ------------------------------
# SIMPLE FALLBACK SENTIMENT
# ------------------------------
def simple_sentiment(text):
    positive_words = ["gain", "growth", "profit", "rise", "positive"]
    negative_words = ["loss", "drop", "fall", "negative", "decline"]

    text = text.lower()

    if any(w in text for w in positive_words):
        return "POSITIVE"
    elif any(w in text for w in negative_words):
        return "NEGATIVE"
    return "NEUTRAL"

# ------------------------------
# STOCK LIST
# ------------------------------
stock_list = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "NFLX",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"
]

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("🔍 Controls")

selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
custom_stock = st.sidebar.text_input("Or Enter Symbol")

stock_symbol = custom_stock if custom_stock else selected_stock

interval = st.sidebar.selectbox(
    "Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
)

# Period logic
if interval == "1m":
    period = "7d"
elif interval in ["5m", "15m", "30m"]:
    period = "60d"
elif interval == "1h":
    period = "1y"
else:
    period = "1y"

# ------------------------------
# FETCH DATA
# ------------------------------
@st.cache_data(ttl=300)
def get_data(symbol, period, interval):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        news = stock.news if stock.news else []
        return hist, news, None
    except Exception as e:
        return None, None, str(e)

hist, news, error = get_data(stock_symbol, period, interval)

# ------------------------------
# FALLBACK DATA
# ------------------------------
if error or hist is None or hist.empty:
    st.warning("⚠️ API limit reached. Showing demo data.")

    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    hist = pd.DataFrame({
        "Open": np.random.rand(100) * 100 + 100,
        "High": np.random.rand(100) * 100 + 120,
        "Low": np.random.rand(100) * 100 + 90,
        "Close": np.random.rand(100) * 100 + 100,
    }, index=dates)

    news = [
        {"title": f"{stock_symbol} shows stable growth outlook"},
        {"title": f"Investors cautious about {stock_symbol}"},
        {"title": f"Market trends impact {stock_symbol}"}
    ]

# ------------------------------
# HEADER
# ------------------------------
st.title("📊 Gen AI Financial Analyst")
st.caption("Real-time stock insights with AI analysis")

# ------------------------------
# CANDLESTICK
# ------------------------------
st.subheader(f"{stock_symbol} Candlestick Chart ({interval})")

fig = go.Figure(data=[go.Candlestick(
    x=hist.index,
    open=hist["Open"],
    high=hist["High"],
    low=hist["Low"],
    close=hist["Close"]
)])

fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, width="stretch")

# ------------------------------
# RSI
# ------------------------------
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(window).mean() / loss.rolling(window).mean()
    return 100 - (100 / (1 + rs))

hist["RSI"] = compute_rsi(hist["Close"])

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], name="RSI"))
fig_rsi.update_layout(template="plotly_dark", height=300)

st.subheader("📊 RSI Indicator")
st.plotly_chart(fig_rsi, width="stretch")

# ------------------------------
# MACD
# ------------------------------
def compute_macd(data):
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

hist["MACD"], hist["Signal"] = compute_macd(hist["Close"])

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["Signal"], name="Signal"))
fig_macd.update_layout(template="plotly_dark", height=300)

st.subheader("📉 MACD Indicator")
st.plotly_chart(fig_macd, width="stretch")

# ------------------------------
# METRICS (FIXED)
# ------------------------------
if hist is not None and not hist.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest", f"${hist['Close'].iloc[-1]:.2f}")
    col2.metric("High", f"${hist['High'].max():.2f}")
    col3.metric("Low", f"${hist['Low'].min():.2f}")

# ------------------------------
# NEWS + SENTIMENT (PARAGRAPH)
# ------------------------------
st.subheader("📰 Market Insight")

titles = []
sentiments = []

# fallback news if empty
if not news:
    news = [
        {"title": f"{stock_symbol} shows steady growth"},
        {"title": f"Investors cautious about {stock_symbol}"}
    ]

for article in news[:5]:
    title = article if isinstance(article, str) else article.get("title", "")
    if not title:
        continue

    titles.append(title)

    try:
        if sentiment_pipeline:
            result = sentiment_pipeline(title[:512])[0]
            label = result["label"]
        else:
            label = simple_sentiment(title)
    except:
        label = simple_sentiment(title)

    sentiments.append(label)

# ------------------------------
# AI PARAGRAPH SUMMARY
# ------------------------------
if titles:
    pos = sentiments.count("POSITIVE")
    neg = sentiments.count("NEGATIVE")

    if pos > neg:
        overall = "positive"
        recommendation = "BUY"
    elif neg > pos:
        overall = "negative"
        recommendation = "SELL"
    else:
        overall = "neutral"
        recommendation = "HOLD"

    summary_text = (
        f"Recent news about {stock_symbol} indicates a {overall} market sentiment. "
        f"Key developments include: {titles[0]}. "
        f"Overall investor outlook appears {overall}, suggesting a {recommendation} strategy "
        f"based on current trends and sentiment analysis."
    )

    st.info(summary_text)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit + Yahoo Finance + NLP")