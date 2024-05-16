# Required Libraries
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import json

# 1. Data Collection and Preprocessing
def fetch_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    print("API Response:")
    print(data)
    print("API Response:")
    print(json.dumps(data, indent=4))

    if 'Time Series (Daily)' in data:
        return data['Time Series (Daily)']
    else:
        raise KeyError(f"'Time Series (Daily)' not found in the response. Response received: {data}")


def clean_stock_data(data):
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric)
    df = df.dropna()
    return df

# 2. Exploratory Data Analysis (EDA)
def visualize_time_series(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['4. close'], label='Close Price')
    plt.title('Stock Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def calculate_statistics(data):
    return data.describe()

def analyze_correlation(data):
    correlation_matrix = data.corr()
    return correlation_matrix

# 4. Model Development
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# 6. Visualization and Interpretability
def visualize_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label='Actual Price')
    plt.plot(y_test.index, y_pred, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main function to run the project
def main():
    # Data Collection and Preprocessing
    symbol = 'GOOG'  # Example stock symbol
    api_key = 'IQT4YV8NCZ6MF9HO'  # Your Alpha Vantage API key
    data = fetch_stock_data(symbol, api_key)
    cleaned_data = clean_stock_data(data)

    # Exploratory Data Analysis (EDA)
    visualize_time_series(cleaned_data)
    statistics = calculate_statistics(cleaned_data)
    print("Descriptive Statistics:")
    print(statistics)
    # correlation_matrix = analyze_correlation(cleaned_data)
    # print("\nCorrelation Matrix:")
    # print(correlation_matrix)
    def analyze_correlation(data):
        correlation_matrix = data.corr()
        return correlation_matrix

    # Feature Engineering (if needed)
    # ...

    # Feature Selection (if needed)
    # ...

    # Splitting Data into Features and Target
    X = cleaned_data.drop(columns=['4. close'])
    y = cleaned_data['4. close']

    # Splitting Data into Train and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Development
    model = train_model(X_train, y_train)

    # Model Evaluation
    mae, mse, rmse = evaluate_model(model, X_test, y_test)
    print("\nModel Evaluation Metrics:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Visualization and Interpretability
    y_pred = model.predict(X_test)
    visualize_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
