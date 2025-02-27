# stock-price-prediction-app
 the project was about a recommendation system using ml, The project name was "stock price prediction using Machine learning". 
 Overview

This Stock Price Prediction App is built using Streamlit, Pandas, Scikit-Learn, and Matplotlib. It allows users to select stock data for Tesla, Reliance, and AAPL, preprocess the dataset, build a prediction model, visualize insights, and make predictions based on historical stock prices.

Features

Sidebar Navigation: Choose between Tesla, Reliance, and AAPL.

Data Preprocessing: Handles missing values and provides dataset insights.

Model Building: Uses Linear Regression to predict stock prices.

Visualization: Generates line plots and bar charts for actual vs. predicted values.

Prediction: Allows users to input stock parameters and get a predicted closing price.

Background Image: Custom styling with a stock market-themed background.

Embedded Video: An educational YouTube video for additional insights.

Installation

Clone the Repository (if applicable)

git clone <repo-link>
cd stock-price-prediction-app

Install Dependencies

pip install -r requirements.txt

Run the App

streamlit run app.py

Dependencies

Make sure you have the following Python libraries installed:

pip install streamlit pandas numpy scikit-learn matplotlib

Project Structure

📂 streamlit-stock-app
│── app.py  # Main application script
│── requirements.txt  # List of dependencies
│── datasets/
│   ├── AAPL.csv
│   ├── TSLA.csv
│   ├── RELIANCE.csv
│── README.md  # Project documentation

Dataset

The dataset used for prediction contains the following columns:

Date: The date of the stock price data.

Open: Opening price.

High: Highest price of the day.

Low: Lowest price of the day.

Close: Closing price.

Volume: Number of shares traded.

Model Details

Algorithm: Linear Regression

Features Used: High, Low, Open, Volume

Target Variable: Close

Evaluation Metrics: MAE, MSE, RMSE

How to Use

Select a stock (Tesla, Reliance, or AAPL) from the sidebar.

Navigate through options (Home, Data Preprocessing, Model Building, Visualization, Prediction).

Visualize stock trends and check for missing values.

Train the model and view coefficients and errors.

Enter stock parameters to predict the closing price.

Download preprocessed data if needed.

Future Improvements

Implement advanced machine learning models like LSTM for better predictions.

Integrate real-time stock market data.

Improve the UI for better user experience.
