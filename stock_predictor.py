import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import timedelta
import matplotlib.pyplot as plt

newsapi = NewsApiClient(api_key = '')

def sentiment(ticker):
    sia = SentimentIntensityAnalyzer()
    today = pd.to_datetime('today').normalize()
    past_date = today - timedelta(days=29)
    all_articles = newsapi.get_everything(q=ticker, from_param=past_date.strftime('%Y-%m-%d'), to=today.strftime('%Y-%m-%d'), language='en', sort_by='publishedAt',page_size=100)
    
    sentiments = []
    for article in all_articles['articles']:
        if article['title']:
            score = sia.polarity_scores(article['title'])['compound']
            sentiments.append({
                'date': pd.to_datetime(article['publishedAt']).normalize(),
                'sentiment': score
            })
    if not sentiments:
        return pd.DataFrame(columns=['date', 'sentiment']).set_index('date')
    
    sentiment_df = pd.DataFrame(sentiments)
    daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean()
    daily_sentiment.index = daily_sentiment.index.tz_localize(None)

    return daily_sentiment

ticker_input = input("Please enter the stock ticker that you want to analyze: \n")
ticker = ticker_input.upper()

stock_data = yf.download(ticker, start="2020-01-01")

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.droplevel(1)

if isinstance(stock_data.index, pd.MultiIndex):
    stock_data.reset_index(inplace=True)
    stock_data.set_index('Date', inplace=True)

stock_data.index = stock_data.index.tz_localize(None)

sentiment_scores = sentiment(ticker)
stock_data = stock_data.join(sentiment_scores)
stock_data.rename(columns={'sentiment': 'sentiment_score'}, inplace=True)
stock_data['sentiment_score'].fillna(0, inplace=True)

stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data.dropna(inplace=True)   
stock_data['Target'] = stock_data['Close'].shift(-1)
stock_data['Lag1_Close'] = stock_data['Close'].shift(1)
stock_data['Lag2_Close'] = stock_data['Close'].shift(2)
stock_data['Lag1_Volume'] = stock_data['Volume'].shift(1)
stock_data.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'Lag1_Close', 'Lag2_Close', 'Lag1_Volume', 'sentiment_score']
target = 'Target'

X = stock_data[features]
y = stock_data[target]

split_point = int(len(X) * 0.9)
X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

latest_data = X.iloc[-1:]
tomorrow_prediction = model.predict(latest_data)
predicted_price = tomorrow_prediction[0]
print(f"The model predicts the closing price for {ticker} tomorrow will be: ${predicted_price:.2f}")

results_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': predictions})
plt.figure(figsize=(14, 7))
plt.plot(results_df['Actual Price'], label='Actual Price', color='blue')
plt.plot(results_df['Predicted Price'], label='Predicted Price', color='red', linestyle='--')
plt.title(f'Stock Price Prediction for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()