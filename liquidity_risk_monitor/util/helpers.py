import os
import pandas as pd
import datetime
import json
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

load_dotenv()

def load_logs():
    logs = []
    log_path = "logs"
    if os.path.exists(log_path):
        for file in os.listdir(log_path):
            if file.endswith(".json"):
                with open(os.path.join(log_path, file), "r") as f:
                    try:
                        logs.append(json.load(f))
                    except Exception:
                        continue
    return logs

def load_data():
    base_path = "data/"
    return {
        "bank_data": pd.read_csv(base_path + "bank_data.csv").to_dict(orient="records"),
        "fixed_payments": pd.read_csv(base_path + "fixed_payments.csv").to_dict(orient="records"),
        "transactions": pd.read_csv(base_path + "transactions_updated.csv"),
        "past_logs": load_logs()
    }

def forecast_with_arima(transactions_df):
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["currency"] == "INR"]
    df["amount"] = df["amount"].astype(float)

    daily_cashflow = df.groupby("date")["amount"].sum().asfreq("D").fillna(0)
    model = ARIMA(daily_cashflow, order=(2, 1, 2))
    model_fit = model.fit()
    forecast_steps = 10
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=daily_cashflow.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_series = pd.Series(forecast, index=forecast_dates)

    return daily_cashflow, forecast_series

def plot_forecast(daily_cashflow, forecast_series):
    plt.figure(figsize=(12, 6))
    plt.plot(daily_cashflow[-30:], label="Recent Cashflow")
    plt.plot(forecast_series, label="Forecast", color="red", linestyle="--")
    plt.title("10-Day Cash Flow Forecast")
    plt.xlabel("Date")
    plt.ylabel("Amount (INR)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
