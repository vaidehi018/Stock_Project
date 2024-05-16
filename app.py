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
from stock import fetch_stock_data, clean_stock_data, visualize_time_series, calculate_statistics, analyze_correlation, train_model, evaluate_model, visualize_predictions


# Import functions from stock.py
from stock import fetch_stock_data, clean_stock_data, visualize_time_series, calculate_statistics, analyze_correlation, train_model, evaluate_model, visualize_predictions

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
    correlation_matrix = analyze_correlation(cleaned_data)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

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

    # Visualize Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    main()
