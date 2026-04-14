import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import pandas as pd
import numpy as np

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Gen AI Financial Analyst", layout="wide")

# ------------------------------
# SENTIMENT MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("🔍 Controls")

stock_symbol = st.sidebar.text_input("Search Stock Symbol", "AAPL")

interval = st.sidebar.selectbox(
    "Interval",
    ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
)

# Auto period handling
if interval == "1m":
    period = "7d"
elif interval in ["5m", "15m", "30m"]:
    period = "60d"
elif interval == "1h":
    period = "1y"
else:
    period = "1y"

# ------------------------------
# FETCH DATA (WITH RATE LIMIT FIX)
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
# FALLBACK DATA (IMPORTANT)
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

    news = ["Market stable", "Investors cautious"]

# ------------------------------
# HEADER
# ------------------------------
st.title("📊 Gen AI Financial Analyst")
st.caption("Real-time stock analysis with AI insights")

# ------------------------------
# CANDLESTICK CHART
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

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# RSI CALCULATION
# ------------------------------
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

hist["RSI"] = compute_rsi(hist["Close"])

# ------------------------------
# MACD CALCULATION
# ------------------------------
def compute_macd(data):
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

hist["MACD"], hist["Signal"] = compute_macd(hist["Close"])

# ------------------------------
# RSI CHART
# ------------------------------
st.subheader("📊 RSI Indicator")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], name="RSI"))

fig_rsi.update_layout(template="plotly_dark", height=300)
st.plotly_chart(fig_rsi, use_container_width=True)

# ------------------------------
# MACD CHART
# ------------------------------
st.subheader("📉 MACD Indicator")

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["Signal"], name="Signal"))

fig_macd.update_layout(template="plotly_dark", height=300)
st.plotly_chart(fig_macd, use_container_width=True)

# ------------------------------
# METRICS
# ------------------------------
if not hist.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest", f"${hist['Close'][-1]:.2f}")
    col2.metric("High", f"${hist['High'].max():.2f}")
    col3.metric("Low", f"${hist['Low'].min():.2f}")

# ------------------------------
# NEWS + SENTIMENT
# ------------------------------
st.subheader("📰 News Sentiment")

titles = []
sentiments = []

for article in news[:5]:
    title = article if isinstance(article, str) else article.get("title", "")
    titles.append(title)

    result = sentiment_pipeline(title)[0]
    sentiments.append(result["label"])

    if result["label"] == "POSITIVE":
        st.success(f"🟢 {title}")
    else:
        st.error(f"🔴 {title}")

# ------------------------------
# SUMMARY
# ------------------------------
st.subheader("🤖 AI Summary")

if titles:
    st.info(" ".join(titles[:2]))

# ------------------------------
# RECOMMENDATION
# ------------------------------
st.subheader("📊 Recommendation")

if sentiments:
    pos = sentiments.count("POSITIVE")
    neg = sentiments.count("NEGATIVE")

    if pos > neg:
        st.success("📈 BUY")
    elif neg > pos:
        st.error("📉 SELL")
    else:
        st.warning("⚖️ HOLD")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit + Yahoo Finance + NLP")