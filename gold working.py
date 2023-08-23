import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import requests

# Fetch Shiba Inu price data from CoinGecko API
url = 'https://api.coingecko.com/api/v3/coins/shiba-inu/market_chart'
params = {
    'vs_currency': 'usd',
    'days': '365',  # Adjust the number of days as per your requirement
}
response = requests.get(url, params=params)
data = response.json()

# Extract the date and price data from the API response
timestamps = data['prices']
prices = [timestamp[1] for timestamp in timestamps]
dates = [pd.to_datetime(timestamp[0], unit='ms') for timestamp in timestamps]

# Create a dataframe with the date and price data
shib = pd.DataFrame({'ds': dates, 'y': prices})

# Create and fit the Prophet model
model = Prophet(daily_seasonality=True)  # Add other relevant parameters if needed
model.fit(shib)

# Generate future dates
start_date = pd.to_datetime('2022-05-12')
end_date = pd.to_datetime('2023-05-20')
future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Make predictions for the future dates
future_predictions = model.predict(pd.DataFrame({'ds': future_dates}))

# Extract the predicted values
predicted_prices = future_predictions['yhat'].values

# Plot the actual and predicted prices
plt.plot(shib['ds'], shib['y'], label='Actual')
plt.plot(future_dates, predicted_prices, label='Predicted')

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Shiba Inu Price Prediction')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Add a legend
plt.legend()

# Show the plot
plt.show()
