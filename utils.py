import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarimax_forecast(series, steps=4):
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 5))
    fit = model.fit(disp=False)
    return fit.forecast(steps).tolist()


def forecast_sales(df, product_input):
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

    preds = sarimax_forecast(sales, 4)

    plt.figure(figsize=(12, 5))

    plt.plot(
        weekly['weekstart'], weekly['sales'],
        color="blue", linewidth=2, marker="o",
        label="Historical Sales"
    )

    last_x = weekly['weekstart'].iloc[-1]
    last_y = weekly['sales'].iloc[-1]
    plt.scatter(last_x, last_y, color="blue", s=35, zorder=5)

    plt.plot(
        [last_x] + future_dates,
        [last_y] + preds,
        color="purple", linewidth=2, marker="o",
        label="SARIMAX Forecast (4 Weeks)"
    )

    plt.title(
        f"Short Term Sales Forecast for Product: {pdf['description'].iloc[0]} using SARIMAX model", fontsize=14)
    plt.xlabel("Week")
    plt.ylabel("Sales")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return preds, future_dates
