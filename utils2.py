import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ============= XGBOOST FORECAST =============


def xgboost_forecast(series, steps=4, window=4):

    s = np.array(series)
    X, y = [], []

    # Create sliding windows (last 4 → next)
    for i in range(window, len(s)):
        X.append(s[i-window:i])
        y.append(s[i])

    X = np.array(X)
    y = np.array(y)

    # Train XGBoost Regressor
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X, y)

    history = list(s)
    preds = []

    # Predict next 4 points
    for _ in range(steps):
        last_window = np.array(history[-window:])
        next_val = model.predict(last_window.reshape(1, -1))[0]
        preds.append(np.float64(next_val))
        history.append(next_val)

    return preds


# ============= MAIN FORECAST FUNCTION =============
def forecast_sales(df, product_input):

    # Filter product
    if product_input in df['stockcode'].astype(str).unique():
        pdf = df[df['stockcode'].astype(str) == product_input]
    else:
        pdf = df[df['description'].str.contains(
            product_input, case=False, na=False)]

    if pdf.empty:
        print("No product found.")
        return

    pdf['weekstart'] = pdf['invoicedate'].dt.to_period(
        "W").apply(lambda r: r.start_time)
    weekly = pdf.groupby('weekstart')['sales'].sum().reset_index()
    weekly = weekly.set_index('weekstart').asfreq(
        "W-MON", fill_value=0).reset_index()

    sales = weekly['sales'].values
    last_date = weekly['weekstart'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(4)]

    preds = xgboost_forecast(sales, 4)

    # ============= PLOT =============
    plt.figure(figsize=(14, 6))

    # Historical
    plt.plot(
        weekly['weekstart'], weekly['sales'],
        color="blue", linewidth=2, marker="o",
        label="Historical Sales"
    )

    last_x = weekly['weekstart'].iloc[-1]
    last_y = weekly['sales'].iloc[-1]
    plt.scatter(last_x, last_y, color="blue", s=35, zorder=5)

    # XGBoost Forecast
    plt.plot(
        [last_x] + future_dates,
        [last_y] + preds,
        color="red", linewidth=2, marker="o",
        label="XGBoost Forecast (4 Weeks)"
    )

    plt.title(
        f"Long Term Sales Forecast for Product: {pdf['description'].iloc[0]} using XGBoost model", fontsize=15)
    plt.xlabel("Week")
    plt.ylabel("Sales")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return preds, future_dates
