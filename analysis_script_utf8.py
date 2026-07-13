
import numpy as np
import pandas as pd


df = df = pd.read_csv('data/train.csv')

df.columns
df.head()

df.info()

df.shape

df.isnull().sum()

df.duplicated().sum()

df[df['Postal Code'].isnull()]

df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

df.head()

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Week'] = df['Order Date'].dt.isocalendar().week
df['DayOfWeek'] = df['Order Date'].dt.dayofweek
df['dayname'] = df['Order Date'].dt.day_name()
df['Quarter'] = df['Order Date'].dt.quarter

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['Month'].apply(get_season)

df.sample()

daily_sales = (
    df.groupby('Order Date', as_index=False)['Sales']
      .sum()
)

daily_sales.head()
weekly_sales = (
    df.resample('W', on='Order Date')['Sales']
      .sum()
      .reset_index()
)

weekly_sales.head()
monthly_sales = (
    df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales']
      .sum()
      .reset_index()
)

monthly_sales.head()
region_yearly_sales = (
    df.groupby(['Region', 'Year'])['Sales']
      .sum()
      .reset_index()
)

region_yearly_sales.head()
monthly_sales = (
    df.resample('ME', on='Order Date')['Sales']
      .sum()
      .reset_index()
)

monthly_sales.head()

import matplotlib.pyplot as plt

category_sales = (
    df.groupby('Category')['Sales']
      .sum()
      .sort_values(ascending=False)
)

print(category_sales)

plt.figure(figsize=(8,5))

category_sales.plot(kind='bar')

plt.title("Total Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")

plt.xticks(rotation=0)

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

sns.lineplot(
    data=region_yearly_sales,
    x='Year',
    y='Sales',
    hue='Region',
    marker='o'
)

plt.title("Yearly Sales Trend by Region")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.grid(True)

plt.show()

df['Shipping Days'] = (
    df['Ship Date'] - df['Order Date']
).dt.days
df[['Order Date','Ship Date','Shipping Days']].head()
shipping_time = (
    df.groupby('Region')['Shipping Days']
      .mean()
      .sort_values()
)

print(shipping_time)
plt.figure(figsize=(8,5))

shipping_time.plot(
    kind='bar',
    color='orange'
)

plt.title("Average Shipping Time by Region")
plt.xlabel("Region")
plt.ylabel("Average Days")

plt.xticks(rotation=0)

plt.grid(axis='y')

plt.show()

monthly_pattern = (
    df.groupby(['Year','Month'])['Sales']
      .sum()
      .reset_index()
)

monthly_pattern.head()
plt.figure(figsize=(12,6))

sns.lineplot(
    data=monthly_pattern,
    x='Month',
    y='Sales',
    hue='Year',
    marker='o'
)

plt.title("Monthly Sales Pattern Across Years")

plt.xticks(range(1,13))

plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)

plt.show()

plt.figure(figsize=(12,6))

sns.lineplot(
    data=monthly_pattern,
    x='Month',
    y='Sales',
    hue='Year',
    marker='o'
)

plt.title("Monthly Sales Pattern Across Years")

plt.xticks(range(1,13))

plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)

plt.show()




!pip install statsmodels
!pip install --upgrade pip
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import seaborn as sns

monthly_sales = (
    df.groupby('Order Date')['Sales']
      .sum()
      .resample('ME')
      .sum()
)
monthly_sales.head()

plt.figure(figsize=(14,6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    linewidth=2
)

plt.title("Monthly sales trend (2014-17)")
plt.title("Monthly Sales Trend (2014û2017)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)

plt.show()

decomposition = seasonal_decompose(
    monthly_sales,
    model = 'addictive',
    period=12
)


fig = decomposition.plot()

fig.set_size_inches(14,10)

plt.show()


result = adfuller(monthly_sales)

print("ADF Statistics :- ", result[0])
print("P-value :- ", result[1])
print("Critical values :- ")

for key , value in result[4].items() :
    print(key, ":", value)
if result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is non-stationary.")


monthly_sales_diff = monthly_sales.diff().dropna()

plt.figure(figsize=(14,5))

plt.plot(
    monthly_sales_diff.index,
    monthly_sales_diff.values
)

plt.title("Differenced Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Differenced Sales")
plt.grid(True)

plt.show()

result_diff = adfuller(monthly_sales_diff)

print("ADF Statistic :", result_diff[0])
print("p-value :", result_diff[1])

for key, value in result_diff[4].items():
    print(key, ":", value)



from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

import numpy as np
import matplotlib.pyplot as plt

monthly_sales = (
    df.groupby('Order Date')['Sales']
      .sum()
      .resample('ME')
      .sum()
)

train = monthly_sales[:-3]
test = monthly_sales[-3:]

print("Training Samples :", len(train))
print("Testing Samples :", len(test))

(p,d,q) = (1,1,1)

(P,D,Q,m) = (1,1,1,12)
sarima_model = SARIMAX(
    train,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_result = sarima_model.fit()

print(sarima_result.summary())

forecast = sarima_result.get_forecast(steps=3)

forecast_values = forecast.predicted_mean

confidence_intervals = forecast.conf_int()
mae = mean_absolute_error(test, forecast_values)

rmse = np.sqrt(
    mean_squared_error(test, forecast_values)
)

mape = mean_absolute_percentage_error(
    test,
    forecast_values
)

print("MAE :", mae)
print("RMSE :", rmse)
print("MAPE :", mape)


plt.figure(figsize=(12,6))

plt.plot(
    train.index,
    train,
    label="Training"
)

plt.plot(
    test.index,
    test,
    label="Actual",
    linewidth=3
)

plt.plot(
    forecast_values.index,
    forecast_values,
    label="Forecast",
    linewidth=3
)

plt.fill_between(
    confidence_intervals.index,
    confidence_intervals.iloc[:,0],
    confidence_intervals.iloc[:,1],
    alpha=0.2
)

plt.title("SARIMA Forecast")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()

final_model = SARIMAX(
    monthly_sales,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_result = final_model.fit()

future_forecast = final_result.get_forecast(steps=3)

future_values = future_forecast.predicted_mean

future_ci = future_forecast.conf_int()

print(future_values)

plt.figure(figsize=(12,6))

plt.plot(
    monthly_sales.index,
    monthly_sales,
    label="Historical Sales"
)

plt.plot(
    future_values.index,
    future_values,
    color="red",
    linewidth=3,
    label="3-Month Forecast"
)

plt.fill_between(
    future_ci.index,
    future_ci.iloc[:,0],
    future_ci.iloc[:,1],
    alpha=0.3
)

plt.title("Future Sales Forecast (SARIMA)")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()



!pip install prophet

from prophet import Prophet

prophet_df = monthly_sales.reset_index()

prophet_df.columns = ['ds', 'y']

prophet_df.head()
train_prophet = prophet_df[:-3]
test_prophet = prophet_df[-3:]

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(train_prophet)

future = model.make_future_dataframe(
    periods=3,
    freq='ME'
)

forecast = model.predict(future)

forecast.tail()

predictions = forecast[['ds', 'yhat']].tail(3)

predictions

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

import numpy as np

mae_prophet = mean_absolute_error(
    test_prophet['y'],
    predictions['yhat']
)

rmse_prophet = np.sqrt(
    mean_squared_error(
        test_prophet['y'],
        predictions['yhat']
    )
)

mape_prophet = mean_absolute_percentage_error(
    test_prophet['y'],
    predictions['yhat']
)

print("MAE :", mae_prophet)
print("RMSE :", rmse_prophet)
print("MAPE :", mape_prophet)
fig = model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()
fig2 = model.plot_components(forecast)
plt.show()


from xgboost import XGBRegressor

xgb_df = monthly_sales.reset_index()

xgb_df.columns = ['Date', 'Sales']

for lag in range(1, 13):
    xgb_df[f'lag_{lag}'] = xgb_df['Sales'].shift(lag)

xgb_df = xgb_df.dropna()

xgb_df.head()

train = xgb_df.iloc[:-3]
test = xgb_df.iloc[-3:]

X_train = train.drop(columns=['Date', 'Sales'])
y_train = train['Sales']

X_test = test.drop(columns=['Date', 'Sales'])
y_test = test['Sales']

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, predictions)

rmse_xgb = np.sqrt(
    mean_squared_error(y_test, predictions)
)

mape_xgb = mean_absolute_percentage_error(
    y_test,
    predictions
)

print("MAE :", mae_xgb)
print("RMSE :", rmse_xgb)
print("MAPE :", mape_xgb)

plt.figure(figsize=(10,5))

plt.plot(
    test['Date'],
    y_test,
    marker='o',
    linewidth=3,
    label='Actual'
)

plt.plot(
    test['Date'],
    predictions,
    marker='o',
    linewidth=3,
    label='Predicted'
)

plt.title("XGBoost Forecast")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()

comparison_df = pd.DataFrame({
    'Model': ['SARIMA', 'Prophet', 'XGBoost'],
    'MAE': [mae, mae_prophet, mae_xgb],
    'RMSE': [rmse, rmse_prophet, rmse_xgb],
    'MAPE': [mape, mape_prophet, mape_xgb]
})

comparison_df = comparison_df.sort_values(by='MAPE')

comparison_df
comparison_df.to_csv("model_comparison.csv", index=False)




def sarima_forecast(data, title):
    """
    Trains a SARIMA model and forecasts the next 3 months.
    """

    # Create monthly sales series
    monthly = (
        data.groupby('Order Date')['Sales']
        .sum()
        .resample('ME')
        .sum()
    )

    # Train model
    model = SARIMAX(
        monthly,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)

    # Forecast
    forecast = result.get_forecast(steps=3)

    future = forecast.predicted_mean

    confidence = forecast.conf_int()

    # Plot
    plt.figure(figsize=(12,5))

    plt.plot(monthly.index,
             monthly.values,
             label="Historical Sales",
             linewidth=2)

    plt.plot(future.index,
             future.values,
             color="red",
             marker='o',
             linewidth=3,
             label="Forecast")

    plt.fill_between(
        confidence.index,
        confidence.iloc[:,0],
        confidence.iloc[:,1],
        alpha=0.2
    )

    plt.title(title)

    plt.xlabel("Date")

    plt.ylabel("Sales")

    plt.legend()

    plt.grid(True)

    plt.show()

    print("\nForecast for Next 3 Months")

    display(future.to_frame(name="Forecast Sales"))

    return future
technology = df[df['Category']=="Technology"]

technology_forecast = sarima_forecast(
    technology,
    "Technology Sales Forecast"
)

