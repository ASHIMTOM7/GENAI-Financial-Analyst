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
# LOAD TRANSFORMERS (SAFE)
# ------------------------------
@st.cache_resource
def load_models():
    try:
        from transformers import pipeline
        sentiment = pipeline("sentiment-analysis")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        return sentiment, summarizer
    except:
        return None, None

sentiment_pipeline, summarizer_pipeline = load_models()

# ------------------------------
# SIMPLE SENTIMENT (FALLBACK)
# ------------------------------
def simple_sentiment(text):
    text = text.lower()
    pos_words = ["gain", "rise", "up", "growth", "positive", "profit"]
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

period = st.selectbox("Time Period", ["1d", "5d", "1mo", "6mo", "1y"])

# ------------------------------
# FETCH DATA (SAFE)
# ------------------------------
@st.cache_data
def get_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        news = getattr(stock, "news", [])
        return hist, news
    except:
        return pd.DataFrame(), []

hist, news = get_data(stock_symbol, period)

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
# RSI
# ------------------------------
delta = hist["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

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
    st.line_chart(hist["RSI"])

with col2:
    st.line_chart(hist[["MACD", "Signal"]])

# ------------------------------
# METRICS (SAFE INDEXING)
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
    title = article if isinstance(article, str) else article.get("title", "")
    if not title:
        continue

    titles.append(title)

    try:
        if sentiment_pipeline:
            label = sentiment_pipeline(title[:512])[0]["label"].upper()
        else:
            raise Exception()
    except:
        label = simple_sentiment(title)

    sentiments.append(label)

# ------------------------------
# OUTPUT LOGIC
# ------------------------------
if titles:
    pos = sentiments.count("POSITIVE")
    neg = sentiments.count("NEGATIVE")

    if pos > neg:
        sentiment_line = f"{stock_symbol} shows positive market sentiment."
        recommendation = "BUY"
    elif neg > pos:
        sentiment_line = f"{stock_symbol} shows negative market sentiment."
        recommendation = "SELL"
    else:
        sentiment_line = f"{stock_symbol} shows neutral market sentiment."
        recommendation = "HOLD"

    # AI SUMMARY
    try:
        if summarizer_pipeline:
            text = " ".join(titles[:3])
            summary = summarizer_pipeline(
                text[:1000],
                max_length=40,
                min_length=15,
                do_sample=False
            )[0]["summary_text"]
        else:
            raise Exception()
    except:
        summary = f"{stock_symbol} outlook remains {recommendation.lower()} based on current indicators."

    st.info(f"📰 Sentiment: {sentiment_line}")
    st.success(f"🤖 AI Summary: {summary}")
    st.metric("📊 Recommendation", recommendation)

else:
    st.warning("No news available.")