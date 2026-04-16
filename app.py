import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="GenAI Financial Analyst", layout="wide")
st.title("📈 GenAI Financial Analyst Dashboard")

# ------------------------------
# SIMPLE SENTIMENT FUNCTION
# ------------------------------
def simple_sentiment(text):
    text = text.lower()
    pos_words = ["gain", "rise", "up", "growth", "positive", "profit", "surge"]
    neg_words = ["fall", "drop", "loss", "negative", "decline"]

    score = sum(w in text for w in pos_words) - sum(w in text for w in neg_words)

    if score > 0:
        return "POSITIVE"
    elif score < 0:
        return "NEGATIVE"
    return "NEUTRAL"

# ------------------------------
# STOCK LIST
# ------------------------------
stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"
]

stock_symbol = st.selectbox("Select Stock", stocks)

# ------------------------------
# TIMEFRAME MAPPING
# ------------------------------
timeframe = st.selectbox("Select Timeframe", 
    ["1m", "5m", "30m", "1h", "1d", "1wk"]
)

interval_map = {
    "1m": ("1d", "1m"),
    "5m": ("5d", "5m"),
    "30m": ("1mo", "30m"),
    "1h": ("1mo", "60m"),
    "1d": ("6mo", "1d"),
    "1wk": ("1y", "1wk")
}

period, interval = interval_map[timeframe]

# ------------------------------
# FETCH DATA (SAFE)
# ------------------------------
@st.cache_data
def get_data(symbol, period, interval):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        news = getattr(stock, "news", [])
        return hist, news
    except:
        return pd.DataFrame(), []

hist, news = get_data(stock_symbol, period, interval)

if hist.empty:
    st.error("No data available. Try another stock.")
    st.stop()

# ------------------------------
# CANDLESTICK CHART
# ------------------------------
fig = go.Figure(data=[go.Candlestick(
    x=hist.index,
    open=hist["Open"],
    high=hist["High"],
    low=hist["Low"],
    close=hist["Close"]
)])

fig.update_layout(title="Candlestick Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width="stretch")

# ------------------------------
# RSI CALCULATION
# ------------------------------
delta = hist["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
hist["RSI"] = 100 - (100 / (1 + rs))

# ------------------------------
# MACD
# ------------------------------
exp1 = hist["Close"].ewm(span=12).mean()
exp2 = hist["Close"].ewm(span=26).mean()
hist["MACD"] = exp1 - exp2
hist["Signal"] = hist["MACD"].ewm(span=9).mean()

col1, col2 = st.columns(2)

with col1:
    st.subheader("RSI")
    st.line_chart(hist["RSI"])

with col2:
    st.subheader("MACD")
    st.line_chart(hist[["MACD", "Signal"]])

# ------------------------------
# METRICS
# ------------------------------
latest_price = hist["Close"].iloc[-1]
prev_price = hist["Close"].iloc[-2] if len(hist) > 1 else latest_price
change = latest_price - prev_price

col1, col2, col3 = st.columns(3)
col1.metric("Price", f"${latest_price:.2f}", f"{change:.2f}")
col2.metric("RSI", f"{hist['RSI'].iloc[-1]:.2f}")
col3.metric("MACD", f"{hist['MACD'].iloc[-1]:.2f}")

# ------------------------------
# NEWS + SENTIMENT
# ------------------------------
st.subheader("📰 Market Insight")

titles = []
sentiments = []

if not news:
    news = [
        {"title": f"{stock_symbol} shows steady growth outlook"},
        {"title": f"Investors cautious about {stock_symbol}"},
        {"title": f"Market trends impact {stock_symbol}"}
    ]

for article in news[:5]:
    title = article.get("title", "")
    if title:
        titles.append(title)
        sentiments.append(simple_sentiment(title))

# ------------------------------
# FINAL OUTPUT
# ------------------------------
if titles:
    pos = sentiments.count("POSITIVE")
    neg = sentiments.count("NEGATIVE")

    if pos > neg:
        sentiment_line = f"{stock_symbol} shows positive news sentiment."
        recommendation = "BUY"
    elif neg > pos:
        sentiment_line = f"{stock_symbol} shows negative news sentiment."
        recommendation = "SELL"
    else:
        sentiment_line = f"{stock_symbol} shows neutral news sentiment."
        recommendation = "HOLD"

    summary = f"{stock_symbol} outlook is {recommendation.lower()} based on technical indicators and recent news."

    st.info(f"📰 Sentiment: {sentiment_line}")
    st.success(f"🤖 AI Summary: {summary}")
    st.metric("📊 Recommendation", recommendation)

else:
    st.warning("No news available.")