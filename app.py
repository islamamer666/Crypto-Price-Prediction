# pip install streamlit fbprophet yfinance plotly requests
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from sprophet.plot import plot_plotly
from plotly import graph_objs as go
import requests

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Cryptocurrency Forecast App')

# Function to fetch top 300 cryptocurrencies
@st.cache_data
def fetch_top_coins(limit=300):
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': False
    }
    response = requests.get(url, params=params)
    data = response.json()
    coins = {coin['name']: coin['symbol'].upper() for coin in data}
    return coins

# Fetch top coins
coins = fetch_top_coins()
crypto_options = list(coins.keys())

selected_crypto_name = st.selectbox('Select cryptocurrency for prediction', crypto_options)
selected_crypto_symbol = coins[selected_crypto_name] + '-USD'

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_crypto_symbol)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="crypto_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="crypto_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
