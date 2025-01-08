# stock-market-analysis
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Fetch stock data using yfinance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Preprocessing the data
def preprocess_data(stock_data):
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['day'] = stock_data['Date'].dt.day
    stock_data['month'] = stock_data['Date'].dt.month
    stock_data['year'] = stock_data['Date'].dt.year
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'day', 'month', 'year']]
    return stock_data

# Step 3: Plotting stock price over time
def plot_stock_data(stock_data, ticker):
    plt.figure(figsize=(10,6))
    plt.plot(stock_data['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Step 4: Predicting future stock prices using linear regression
def predict_stock_price(stock_data):
    stock_data['Date'] = pd.to_numeric(stock_data['Date'])
    X = stock_data[['Date']]
    y = stock_data['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future stock prices
    future_dates = np.array([pd.to_numeric(pd.Timestamp("2025-01-01"))]).reshape(-1, 1)
    future_prediction = model.predict(future_dates)
    print(f'Predicted stock price for 2025-01-01: ${future_prediction[0]:.2f}')

# Main execution
if __name__ == "__main__":
    ticker = 'AAPL'  # Example: Apple stock
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Fetch and preprocess stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    processed_data = preprocess_data(stock_data)
    
    # Plot the stock data
    plot_stock_data(processed_data, ticker)
    
    # Predict the stock price for a future date
    predict_stock_price(processed_data)
