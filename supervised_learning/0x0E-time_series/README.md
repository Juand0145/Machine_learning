# 0x0E. Time Series Forecasting

In the following directory we will find a copilation of files wich porpuse is to understand the concepts of using  Time Series Forecasting and help us to answer the next questions.
-   What is time series forecasting?
-   What is a stationary process?
-   What is a sliding window?
-   How to preprocess time series data
-   How to create a data pipeline in tensorflow for time series data
-   How to perform time series forecasting with RNNs in tensorflow

## Files
- README.md
- forecast_btc.py
- preprocess_data.py

Bitcoin (BTC) became a trending topic after its price peaked in 2018. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.

Given the coinbase and bitstamp datasets, write a script, forecast_btc.py, that creates, trains, and validates a keras model for the forecasting of BTC:

Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):
The datasets are formatted such that every row represents a 60 second time window containing:
The start time of the time window in Unix time
The open price in USD at the start of the time window
The high price in USD within the time window
The low price in USD within the time window
The close price in USD at end of the time window
The amount of BTC transacted in the time window
The amount of Currency (USD) transacted in the time window
The volume-weighted average price in USD for the time window
Your model should use an RNN architecture of your choosing
Your model should use mean-squared error (MSE) as its cost function
You should use a tf.data.Dataset to feed data to your model
Because the dataset is raw, you will need to create a script, preprocess_data.py to preprocess this data. Here are some things to consider:

Are all of the data points useful?
Are all of the data features useful?
Should you rescale the data?
Is the current time window relevant?
How should you save this preprocessed data?