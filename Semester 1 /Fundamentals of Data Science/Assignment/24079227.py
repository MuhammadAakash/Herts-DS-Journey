import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------------
# Config
# -------------------------------------
STUDENT_ID = "24079227"
DATA_PATH = "sales2.csv"

# -------------------------------------
# Data Preprocessing
# -------------------------------------
def load_and_prepare_data(path):
    """Loads the CSV file and performs basic preprocessing."""
    df = pd.read_csv(path, parse_dates=['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Add computed columns
    df['TotalItems'] = df[['NumberGroceryShop', 'NumberGroceryOnline',
                           'NumberNongroceryShop', 'NumberNongroceryOnline']].sum(axis=1)
    df['TotalRevenue'] = df[['RevenueGrocery', 'RevenueNongrocery']].sum(axis=1)
    df['AvgPrice'] = df['TotalRevenue'] / df['TotalItems']
    return df

# -------------------------------------
# Fourier Series Approximation
# -------------------------------------
def compute_fourier_series(y, n_terms=8):
    """
    Returns the Fourier series approximation of `y` with `n_terms` terms.
    """
    x = np.arange(len(y))
    N = len(x)
    result = np.ones(N) * np.mean(y)

    for n in range(1, n_terms + 1):
        a_n = 2 / N * np.sum(y * np.cos(2 * np.pi * n * x / N))
        b_n = 2 / N * np.sum(y * np.sin(2 * np.pi * n * x / N))
        result += a_n * np.cos(2 * np.pi * n * x / N) + b_n * np.sin(2 * np.pi * n * x / N)

    return result

# -------------------------------------
# Figure 1: Bar + Fourier Approximation
# -------------------------------------
def plot_avg_items_and_fourier(df, student_id):
    """
    Plots Figure 1:
    - Bar chart for average daily items sold per month
    - Fourier approximation of 2022 daily items sold
    """
    df_2022 = df[df['Year'] == 2022].copy()
    monthly_avg = df.groupby('Month')['TotalItems'].mean()

    y = df_2022['TotalItems'].values
    fourier_approx = compute_fourier_series(y, n_terms=8)

    plt.figure(figsize=(14, 6))

    # Bar chart of monthly averages
    plt.bar(monthly_avg.index, monthly_avg.values, label='Avg Daily Items Sold (Monthly)', color='skyblue')

    # Fourier approximation as daily trend
    month_days = df_2022['Month'].values + (df_2022['Date'].dt.day / 31)
    plt.plot(month_days, fourier_approx, label='Fourier Approx. (8 terms)', color='darkred', linewidth=2)

    # Labels and formatting
    plt.title(f"Figure 1: Monthly Avg Daily Items Sold & Fourier Approx (Student ID: {student_id})", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Average Daily Items Sold")
    plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def annotate_x_y(X, Y):
    """
    Annotates X and Y values on the plot.
    :param X: Value of X
    :param Y: Value of Y
    """
    return plt.text(0.95, 0.05, f"X (2022 Grocery Revenue): {X:.2f}\nY (2022 Grocery Revenue): {Y:.2f}", 
             fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom', 
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# -------------------------------------
# Figure 2: Scatter + Regression + Annotations
# -------------------------------------
def plot_price_vs_items(df, student_id):
    """
    Plots Figure 2:
    - Scatter plot of TotalItems vs AvgPrice
    - Linear regression line
    - X and Y annotations based on student ID ending
    """
    plt.figure(figsize=(10, 6))

    # Scatter
    plt.scatter(df['TotalItems'], df['AvgPrice'], alpha=0.5, label='Data Points')

    # Regression
    X_lin = df['TotalItems'].values.reshape(-1, 1)
    y_lin = df['AvgPrice'].values
    model = LinearRegression().fit(X_lin, y_lin)
    y_pred = model.predict(X_lin)
    plt.plot(df['TotalItems'], y_pred, color='red', label='Linear Regression Fit')

    # Annotation values (for ID ending with 7 → grocery revenue)
    grocery_2022 = df[df['Year'] == 2022]
    X = grocery_2022['RevenueGrocery'].sum()
    Y = X  # Same in this context

    # Annotations
    annotate_x_y(X, Y)

    # Labels
    plt.title(f"Figure 2: Avg Price vs Total Items Sold (Student ID: {student_id})", fontsize=14)
    plt.xlabel("Total Items Sold")
    plt.ylabel("Average Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------
# Run all
# -------------------------------------
def main():
    df = load_and_prepare_data(DATA_PATH)
    plot_avg_items_and_fourier(df, STUDENT_ID)
    plot_price_vs_items(df, STUDENT_ID)

# Execute
if __name__ == "__main__":
    main()
