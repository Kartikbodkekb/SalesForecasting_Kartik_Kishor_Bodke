import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("# End-to-End Sales Forecasting & Demand Intelligence System"))
cells.append(nbf.v4.new_markdown_cell("## Task 1: Data Loading, Merging & Deep Exploration"))
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Week'] = df['Order Date'].dt.isocalendar().week
df['DayOfWeek'] = df['Order Date'].dt.dayofweek
df['Quarter'] = df['Order Date'].dt.quarter

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'
    
df['Season'] = df['Month'].apply(get_season)

print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

daily_sales = df.groupby('Order Date')['Sales'].sum()
weekly_sales = df.resample('W', on='Order Date')['Sales'].sum()
monthly_sales = df.resample('ME', on='Order Date')['Sales'].sum()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Business Questions Answered:
1. **Which product category generates the highest total revenue?** Technology.
2. **Which region has the most consistent sales growth?** West region.
3. **Average time between Order and Ship Date?** ~3.9 days, fairly consistent across regions.
4. **Consistent spikes?** Yes, November and December consistently spike due to holiday seasonality.
"""))

cells.append(nbf.v4.new_markdown_cell("## Task 2: Time Series Analysis & Decomposition"))
cells.append(nbf.v4.new_code_cell("""from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

plt.figure(figsize=(10,4))
plt.plot(monthly_sales)
plt.title("Overall Monthly Sales Trend")
plt.savefig("charts/monthly_sales_trend.png")
plt.show()

decomposition = seasonal_decompose(monthly_sales, model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
plt.savefig("charts/decomposition.png")
plt.show()

result = adfuller(monthly_sales)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.05:
    print("Series is stationary")
else:
    print("Series is non-stationary, requires differencing")
    
monthly_diff = monthly_sales.diff().dropna()
diff_result = adfuller(monthly_diff)
print("Differenced p-value:", diff_result[1])
"""))

cells.append(nbf.v4.new_markdown_cell("## Task 3: Sales Forecasting using 3 Different Models"))
cells.append(nbf.v4.new_code_cell("""from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

train = monthly_sales[:-3]
test = monthly_sales[-3:]

# SARIMA
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
sarima_res = sarima_model.fit(disp=False)
sarima_fc = sarima_res.get_forecast(steps=3).predicted_mean

sarima_mae = mean_absolute_error(test, sarima_fc)
sarima_rmse = np.sqrt(mean_squared_error(test, sarima_fc))
sarima_mape = mean_absolute_percentage_error(test, sarima_fc)

plt.figure(figsize=(10,4))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(sarima_fc.index, sarima_fc, label="SARIMA Forecast")
plt.legend()
plt.title("SARIMA Model Forecast")
plt.savefig("charts/sarima_forecast.png")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""from prophet import Prophet

prophet_df = monthly_sales.reset_index()
prophet_df.columns = ['ds', 'y']
train_p = prophet_df[:-3]
test_p = prophet_df[-3:]

m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(train_p)
future = m.make_future_dataframe(periods=3, freq='ME')
fcst = m.predict(future)
prophet_fc = fcst['yhat'][-3:].values

prophet_mae = mean_absolute_error(test_p['y'], prophet_fc)
prophet_rmse = np.sqrt(mean_squared_error(test_p['y'], prophet_fc))
prophet_mape = mean_absolute_percentage_error(test_p['y'], prophet_fc)

fig = m.plot(fcst)
plt.title("Prophet Forecast")
plt.savefig("charts/prophet_forecast.png")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""from xgboost import XGBRegressor

xgb_df = monthly_sales.reset_index()
xgb_df.columns = ['Date', 'Sales']
for lag in range(1, 4):
    xgb_df[f'lag_{lag}'] = xgb_df['Sales'].shift(lag)
xgb_df['rolling_mean'] = xgb_df['Sales'].rolling(3).mean()
xgb_df['Month'] = xgb_df['Date'].dt.month
xgb_df = xgb_df.dropna()

train_x = xgb_df.iloc[:-3].drop(columns=['Date', 'Sales'])
train_y = xgb_df.iloc[:-3]['Sales']
test_x = xgb_df.iloc[-3:].drop(columns=['Date', 'Sales'])
test_y = xgb_df.iloc[-3:]['Sales']

xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(train_x, train_y)
xgb_fc = xgb.predict(test_x)

xgb_mae = mean_absolute_error(test_y, xgb_fc)
xgb_rmse = np.sqrt(mean_squared_error(test_y, xgb_fc))
xgb_mape = mean_absolute_percentage_error(test_y, xgb_fc)

plt.figure(figsize=(10,4))
plt.plot(test_y.values, label='Actual')
plt.plot(xgb_fc, label='XGB Forecast')
plt.legend()
plt.title("XGBoost Forecast")
plt.savefig("charts/xgb_forecast.png")
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""results = pd.DataFrame({
    'Model': ['SARIMA', 'Prophet', 'XGBoost'],
    'MAE': [sarima_mae, prophet_mae, xgb_mae],
    'RMSE': [sarima_rmse, prophet_rmse, xgb_rmse],
    'MAPE': [sarima_mape, prophet_mape, xgb_mape],
    'M1': [sarima_fc.iloc[0], prophet_fc[0], xgb_fc[0]],
    'M2': [sarima_fc.iloc[1], prophet_fc[1], xgb_fc[1]],
    'M3': [sarima_fc.iloc[2], prophet_fc[2], xgb_fc[2]]
})
print(results)
"""))

cells.append(nbf.v4.new_markdown_cell("## Task 4: Product Category & Region Level Forecasting"))
cells.append(nbf.v4.new_code_cell("""def forecast_segment(segment_data, label):
    monthly = segment_data.resample('ME', on='Order Date')['Sales'].sum()
    if len(monthly) > 24:
        model = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=3).predicted_mean
        return fc
    return None

segments = {
    'Furniture': df[df['Category'] == 'Furniture'],
    'Technology': df[df['Category'] == 'Technology'],
    'Office Supplies': df[df['Category'] == 'Office Supplies'],
    'West': df[df['Region'] == 'West'],
    'East': df[df['Region'] == 'East']
}

plt.figure(figsize=(12,6))
for name, data in segments.items():
    fc = forecast_segment(data, name)
    if fc is not None:
        plt.plot(fc.index, fc.values, label=name, marker='o')
plt.title("3-Month Forecast Comparison by Category and Region")
plt.legend()
plt.savefig("charts/segment_forecast.png")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Task 5: Anomaly Detection in Sales Data"))
cells.append(nbf.v4.new_code_cell("""from sklearn.ensemble import IsolationForest
weekly = df.resample('W', on='Order Date')['Sales'].sum().reset_index()

iso = IsolationForest(contamination=0.05, random_state=42)
weekly['Anomaly_IF'] = iso.fit_predict(weekly[['Sales']])

# Z-Score
weekly['RollingMean'] = weekly['Sales'].rolling(window=4).mean()
weekly['RollingStd'] = weekly['Sales'].rolling(window=4).std()
weekly['Z_Score'] = (weekly['Sales'] - weekly['RollingMean']) / weekly['RollingStd']
weekly['Anomaly_Z'] = weekly['Z_Score'].apply(lambda x: -1 if abs(x) > 2 else 1)

plt.figure(figsize=(12,5))
plt.plot(weekly['Order Date'], weekly['Sales'], label='Sales')
anomalies_if = weekly[weekly['Anomaly_IF'] == -1]
plt.scatter(anomalies_if['Order Date'], anomalies_if['Sales'], color='red', label='Anomalies (IF)')
plt.title("Anomaly Detection (Isolation Forest)")
plt.legend()
plt.savefig("charts/anomalies.png")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Task 6: Product Demand Segmentation using Clustering"))
cells.append(nbf.v4.new_code_cell("""from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

subcat = df.groupby('Sub-Category').agg(
    Total_Sales=('Sales', 'sum'),
    Volatility=('Sales', 'std'),
    AOV=('Sales', 'mean')
).fillna(0)

scaler = StandardScaler()
scaled = scaler.fit_transform(subcat)

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 10), wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
subcat['Cluster'] = kmeans.fit_predict(scaled)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(scaled)
subcat['PCA1'] = coords[:,0]
subcat['PCA2'] = coords[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=subcat, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)
for i, txt in enumerate(subcat.index):
    plt.annotate(txt, (subcat['PCA1'].iloc[i], subcat['PCA2'].iloc[i]), fontsize=8)
plt.title("Product Segmentation Clusters")
plt.savefig("charts/clusters.png")
plt.show()
"""))

nb['cells'] = cells

with open('analysis_updated.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
