import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Gen AI Financial Analyst",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# LOAD MODELS (cached)
# ------------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return sentiment, summarizer

sentiment_pipeline, summarizer_pipeline = load_models()

# ------------------------------
# UI HEADER
# ------------------------------
st.title("📊 Gen AI Financial Analyst")
st.markdown("AI-powered stock analysis with sentiment & summary")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("User Input")

stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y"])

# ------------------------------
# FETCH STOCK DATA
# ------------------------------
@st.cache_data
def get_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    return stock, hist

stock, hist = get_stock_data(stock_symbol, period)

# ------------------------------
# STOCK CHART
# ------------------------------
st.subheader(f"📈 Stock Price: {stock_symbol}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist.index,
    y=hist["Close"],
    mode='lines',
    name='Close Price'
))

fig.update_layout(
    title="Stock Price Trend",
    xaxis_title="Date",
    yaxis_title="Price",
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# KEY METRICS
# ------------------------------
col1, col2, col3 = st.columns(3)

if not hist.empty:
    col1.metric("Latest Price", f"${hist['Close'][-1]:.2f}")
    col2.metric("Highest", f"${hist['Close'].max():.2f}")
    col3.metric("Lowest", f"${hist['Close'].min():.2f}")

# ------------------------------
# NEWS SECTION
# ------------------------------
st.subheader("📰 News & Sentiment Analysis")

news = stock.news

if news:
    sentiments = []
    titles = []

    for article in news[:5]:
        title = article["title"]
        titles.append(title)

        result = sentiment_pipeline(title)[0]
        sentiments.append(result["label"])

        st.markdown(f"**📰 {title}**")
        st.write(f"Sentiment: {result['label']} ({result['score']:.2f})")
        st.write("---")

# ------------------------------
# AI SUMMARY
# ------------------------------
if news:
    st.subheader("🤖 AI Generated Summary")

    combined_text = " ".join(titles)

    if len(combined_text) > 50:
        summary = summarizer_pipeline(
            combined_text,
            max_length=60,
            min_length=20,
            do_sample=False
        )

        st.success(summary[0]["summary_text"])

# ------------------------------
# SIMPLE RECOMMENDATION LOGIC
# ------------------------------
st.subheader("📊 Basic Recommendation")

if sentiments:
    positive = sentiments.count("POSITIVE")
    negative = sentiments.count("NEGATIVE")

    if positive > negative:
        st.success("📈 Recommendation: BUY (Positive sentiment)")
    elif negative > positive:
        st.error("📉 Recommendation: SELL (Negative sentiment)")
    else:
        st.warning("⚖️ Recommendation: HOLD (Neutral sentiment)")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, HuggingFace, and Yahoo Finance")