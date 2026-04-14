import streamlit as st
import yfinance as yf
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
# CUSTOM CSS (PREMIUM UI)
# ------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    color: #00ffcc;
}
.metric-box {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD SENTIMENT MODEL (LIGHT)
# ------------------------------
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

# ------------------------------
# HEADER
# ------------------------------
st.title("📊 Gen AI Financial Analyst")
st.caption("AI-powered stock insights with sentiment analysis")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("🔍 User Input")

stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y"])

# ------------------------------
# FETCH DATA
# ------------------------------
@st.cache_data
def get_data(symbol, period):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    news = stock.news
    return hist, news

hist, news = get_data(stock_symbol, period)

# ------------------------------
# STOCK CHART
# ------------------------------
st.subheader(f"📈 {stock_symbol} Stock Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hist.index,
    y=hist["Close"],
    mode='lines',
    name='Close Price'
))

fig.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# METRICS
# ------------------------------
if not hist.empty:
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Latest Price", f"${hist['Close'][-1]:.2f}")
    col2.metric("📈 High", f"${hist['Close'].max():.2f}")
    col3.metric("📉 Low", f"${hist['Close'].min():.2f}")

# ------------------------------
# NEWS + SENTIMENT
# ------------------------------
st.subheader("📰 News Sentiment Analysis")

news = stock.news
titles = []
sentiments = []

if news:
    for article in news[:5]:
        title = article["title"]
        titles.append(title)

        result = sentiment_pipeline(title)[0]
        sentiments.append(result["label"])

        if result["label"] == "POSITIVE":
            st.success(f"🟢 {title}")
        else:
            st.error(f"🔴 {title}")

# ------------------------------
# AI SUMMARY (LIGHT VERSION)
# ------------------------------
st.subheader("🤖 AI Summary")

if titles:
    summary = " ".join(titles[:2])
    st.info(summary)

# ------------------------------
# RECOMMENDATION
# ------------------------------
st.subheader("📊 Investment Recommendation")

if sentiments:
    pos = sentiments.count("POSITIVE")
    neg = sentiments.count("NEGATIVE")

    if pos > neg:
        st.success("📈 BUY – Positive Market Sentiment")
    elif neg > pos:
        st.error("📉 SELL – Negative Market Sentiment")
    else:
        st.warning("⚖️ HOLD – Neutral Market Condition")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit | Yahoo Finance API | NLP Models")