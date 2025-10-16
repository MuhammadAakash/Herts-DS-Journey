import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Load the dataset ===
df = pd.read_csv('sales7.csv')

# === Create date-related columns ===
start_date = pd.to_datetime('2021-01-01')
df['Date'] = start_date + pd.to_timedelta(df['DayNumber'], unit='D')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfYear'] = df['Date'].dt.dayofyear

# === Calculate quantities and total items sold ===
df['QuantityGrocery'] = df['RevenueGrocery'] / df['PriceGrocery']
df['QuantityNongrocery'] = df['RevenueNongrocery'] / df['PriceNongrocery']
df['TotalItemsSold'] = df['QuantityGrocery'] + df['QuantityNongrocery']

# ----------------------------------------------------
# === Figure 1: Monthly Avg Bar Chart + Fourier Line ===
# ----------------------------------------------------

# Monthly average (from all years)
monthly_avg = df.groupby(df['Date'].dt.month)['TotalItemsSold'].mean()

# Fourier series: using 2022 data
df_2022 = df[df['Year'] == 2022].copy()
t = np.arange(365)
y = df_2022['TotalItemsSold'].values

# 8-term Fourier series approximation
N = 8
a0 = np.mean(y)
fourier_approx = np.full_like(t, a0, dtype=float)

for n in range(1, N + 1):
    an = 2/365 * np.sum(y * np.cos(2 * np.pi * n * t / 365))
    bn = 2/365 * np.sum(y * np.sin(2 * np.pi * n * t / 365))
    fourier_approx += an * np.cos(2 * np.pi * n * t / 365) + bn * np.sin(2 * np.pi * n * t / 365)

# Plot Figure 1
plt.figure(figsize=(12, 6))
plt.bar(range(1, 13), monthly_avg, color='skyblue', label='Monthly Avg Daily Items Sold')
plt.plot((t / 30.5) + 1, fourier_approx, color='red', label='Fourier Approximation (2022)', linewidth=2)
plt.xlabel('Month')
plt.ylabel('Average Daily Items Sold')
plt.title('Figure 1: Monthly Avg Daily Items Sold & Fourier Approximation (Student ID: 23093874)')
plt.xticks(ticks=range(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------
# === Figure 2: Scatter Plot + Regression Line + X/Y ===
# --------------------------------------------

# Compute average daily price
df['TotalRevenue'] = df['RevenueGrocery'] + df['RevenueNongrocery']
df['AvgPrice'] = df['TotalRevenue'] / df['TotalItemsSold']

# Prepare data for regression
X_scatter = df['AvgPrice'].values.reshape(-1, 1)
y_scatter = df['TotalItemsSold'].values

model = LinearRegression()
model.fit(X_scatter, y_scatter)
y_pred = model.predict(X_scatter)

# Student ID ends in 4 → compute % revenue from grocery and non-grocery
total_revenue = df['TotalRevenue'].sum()
grocery_revenue = df['RevenueGrocery'].sum()
nongrocery_revenue = df['RevenueNongrocery'].sum()

X_val = (grocery_revenue / total_revenue) * 100
Y_val = (nongrocery_revenue / total_revenue) * 100

# Plot Figure 2
plt.figure(figsize=(10, 6))
plt.scatter(df['AvgPrice'], df['TotalItemsSold'], alpha=0.5, label='Daily Data')
plt.plot(df['AvgPrice'], y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Average Price per Item')
plt.ylabel('Total Items Sold')
plt.title('Figure 2: Avg Price vs Total Items Sold (Student ID: 23093874)')
plt.legend()
plt.grid(True)

# Annotate with X and Y values
plt.text(0.05 + df['AvgPrice'].min(), df['TotalItemsSold'].max() * 0.95,
         f"X (Grocery % Revenue): {X_val:.2f}%", fontsize=10)
plt.text(0.05 + df['AvgPrice'].min(), df['TotalItemsSold'].max() * 0.90,
         f"Y (Non-Grocery % Revenue): {Y_val:.2f}%", fontsize=10)

plt.tight_layout()
plt.show()
