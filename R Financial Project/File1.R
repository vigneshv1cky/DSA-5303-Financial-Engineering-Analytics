# Load libraries
library(quantmod)
library(forecast)
library(prophet)
library(keras)
library(tensorflow)
library(tseries)
library(TTR)
library(ggplot2)
library(tidyverse)
library(reticulate)

# Get Apple stock data from Yahoo Finance
getSymbols("AAPL", src = "yahoo", from = "2010-01-01", to = "2024-01-01")

# Convert the xts object to a tibble
apple_data <- as_tibble(AAPL)

# Rename columns and ensure the date column is in the correct format
colnames(apple_data) <- c("Open", "High", "Low", "Close", "Volume", "Adjusted")
apple_data$Date <- index(AAPL)
apple_data <- apple_data %>% select(Date, everything())

# Plot Daily Closing prices
ggplot(apple_data, aes(x = Date, y = Close)) +
  geom_line() +
  labs(title = "Apple Inc. (AAPL) Closing Prices",
       x = "Date",
       y = "Closing Price (USD)") 

# Plot Daily Volume
ggplot(apple_data, aes(x = Date, y = Volume)) +
  geom_point(alpha = 0.5) +
  labs(title = "Apple Inc. (AAPL) Closing Prices",
       x = "Date",
       y = "Volume") 

# Mutate Daily Return and Daily Percent Change in apple_data
apple_data <- apple_data %>%
  arrange(Date) %>%
  mutate(
    Daily_Return = Close - lag(Close),
    Daily_Percent_Change = (Close / lag(Close) - 1) * 100
  )

# Replace the first NA value caused by differencing with 0.0000..
apple_data <- apple_data %>%
  mutate(
    Daily_Return = replace_na(Daily_Return, 0),
    Daily_Percent_Change = replace_na(Daily_Percent_Change, 0)
  )

# Mutate SMA for 50, 100 and 200 days in apple_data
apple_data <- apple_data %>%
  arrange(Date) %>%
  mutate(
    MA_50 = SMA(Close, n = 50),
    MA_100 = SMA(Close, n = 100),
    MA_200 = SMA(Close, n = 200)
  )

# Replace the first 50, 100 and 200 NA values respectively
apple_data <- apple_data %>%
  mutate(
    MA_50 = replace_na(MA_50, 0),
    MA_100 = replace_na(MA_100, 0),
    MA_200 = replace_na(MA_200, 0)
  )

# Plot SMA for 50,100 and 200 days
ggplot(apple_data, aes(x = Date)) +
  geom_line(aes(y = Close, color = "Actual"), linewidth = 0.5) +
  geom_line(aes(y = MA_50, color = "50_Moving_Average"), linewidth = 0.7) +
  geom_line(aes(y = MA_100, color = "100_Moving_Average"), linewidth = 0.7) +
  geom_line(aes(y = MA_200, color = "200_Moving_Average"), linewidth = 0.7) +
  labs(title = "Apple Inc. (AAPL) Closing Prices with Moving Averages",
       x = "Date",
       y = "Price (USD)") + 
  scale_y_continuous(labels = scales::dollar) +
  scale_color_manual(values = c("Actual" = "black",
                                "50_Moving_Average" = "green",
                                "100_Moving_Average" = "red",
                                "200_Moving_Average" = "blue")) + 
  guides(color = guide_legend(title = ""))


# Perform ADF test on Close
adf.test(apple_data$Close, alternative = "stationary")

# Perform ADF test on Daily Return
adf.test(apple_data$Daily_Return, alternative = "stationary")

# Perform ADF test on Daily Percent Change
adf.test(apple_data$Daily_Percent_Change, alternative = "stationary")

# Plot Daily Return
ggplot(apple_data, aes(x = Date, y = Daily_Return)) +
  geom_line() +
  labs(title = "Daily Returns of Apple Inc. (AAPL)",
       x = "Date",
       y = "Daily Return (USD)")

# Plot Daily Percent Change
ggplot(apple_data, aes(x = Date, y = Daily_Percent_Change)) +
  geom_line() +
  labs(title = "Daily Percent Change of Apple Inc. (AAPL)",
       x = "Date",
       y = "Daily Percent Change (%)") 

acf(apple_data$Close, main = "Autocorrelation of Closed Prices")
pacf(apple_data$Close, main = "Partial Autocorrelation of Closed Prices")

acf(apple_data$Daily_Return, main = "Autocorrelation of Daily_Return Prices")
pacf(apple_data$Daily_Return, main = "Partial Autocorrelation of Daily_Return Prices")

# Split the dataset into train and test dataset
# train_size <-  floor(0.8 * nrow(apple_data))
# train_data <-  apple_data[1:train_size, ]
# test_data <-  apple_data[(train_size + 1):nrow(apple_data), ]
train_data <- apple_data %>% 
  filter(Date <= "2023-06-01")
test_data <- apple_data %>% 
  filter(Date > "2023-06-01")

train_size <- apple_data %>% 
  filter(Date <= "2023-06-01") %>% 
  nrow()

# Fit an ARIMA model to the differenced series
arima_model <- auto.arima(train_data$Close)

# Print the summary of the ARIMA model
summary(arima_model)

# Fit Arima Model and get its predictions
forecast_result <- forecast(arima_model, h = nrow(test_data))
arima_predictions <- tibble(Date = test_data$Date, 
                            forecast = forecast_result$mean)

# Evaluate the model
arima_rmse <- sqrt(mean((forecast_result$mean - test_data$Close)^2))
print(paste("ARIMA RMSE:", arima_rmse))

# Plot the predictions with the test data
ggplot() +
  geom_line(data = train_data, aes(x = Date, y = Close, 
                                   color = "Training Data"), linewidth = 1) +
  geom_line(data = test_data, aes(x = Date, y = Close, 
                                  color = "Actual Data"), linewidth = 1) +
  geom_line(data = arima_predictions, aes(x = Date, y = forecast,
                                          color = "Predicted Data"), linewidth = 1) +
  labs(title = "ARIMA Model Predictions vs Actual Test Data",
       x = "Date",
       y = "Close Price",
       color = "Legend") +
  scale_color_manual(values = c("Training Data" = "black", 
                                "Actual Data" = "red", 
                                "Predicted Data" = "blue")) +
  guides(color = guide_legend(title = ""))

# Prophet model from Facebook
Prophet_train_data <- apple_data %>% 
  filter(Date <= "2023-06-01") %>% 
  select(Date,Close)

Prophet_test_data <- apple_data %>% 
  filter(Date > "2023-06-01") %>% 
  select(Date,Close)

# Change Column Names as prophet model requires Mandotary "ds" and "y" variables.
colnames(Prophet_train_data) <- c("ds","y")

# Fit the prophet Model
Prophet_model <-  prophet(Prophet_train_data)

# Make future predictions
future <- make_future_dataframe(Prophet_model, periods = nrow(Prophet_test_data))
forecast <- predict(Prophet_model, future)

# Evaluate the model
prophet_forecast <- forecast$yhat[(train_size + 1):nrow(forecast)]
prophet_rmse <- sqrt(mean((prophet_forecast - Prophet_test_data$Close)^2))
print(paste("Prophet RMSE:", prophet_rmse))

# Combine the actual and predicted data for plotting
plot_data <- Prophet_test_data %>%
  rename(actual = Close) %>%
  mutate(predicted = prophet_forecast)

# Plot the predictions with the test data
ggplot() +
  geom_line(data = train_data, aes(x = Date, y = Close, 
                                   color = "Training Data"), linewidth = 1) +
  geom_line(data = plot_data, aes(x = Date, y = actual, 
                                  color = "Actual Data"), linewidth = 1) +
  geom_line(data = plot_data, aes(x = Date, y = predicted,
                                  color = "Predicted Data"), linewidth = 1) +
  labs(title = "Prophet Model Predictions vs Actual Test Data",
       x = "Date",
       y = "Close Price",
       color = "Legend") +
  scale_color_manual(values = c("Training Data" = "black", 
                                "Actual Data" = "red", 
                                "Predicted Data" = "blue")) +
  guides(color = guide_legend(title = ""))
