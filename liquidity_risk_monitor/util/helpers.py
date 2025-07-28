import os
import pandas as pd
import datetime
import json
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_model():
    # Model name can be adjusted depending on your quota
    # ("gemini-1.5-flash" is fastest and cheapest, "gemini-2.0-pro" is maximal quality)
    return genai.GenerativeModel("gemini-2.5-pro")

def gemini_simple_chat(prompt, timeout=60):
    print("Calling Gemini...")
    model = get_gemini_model()
    try:
        response = model.generate_content(prompt)
        print("Gemini complete.")
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return ""

def load_logs():
    logs = []
    log_path = "logs"
    if os.path.exists(log_path):
        for file in os.listdir(log_path):
            if file.endswith(".json") and file.startswith("liquidity_risk_log_"):
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
        "transactions": pd.read_csv(base_path + "transactions_updated.csv").to_dict(orient="records"),
        "logs": load_logs()
    }

def forecast_with_arima(transactions):
    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["currency"] == "INR"]
    df["amount"] = df["amount"].astype(float)
    daily_cashflow = df.groupby("date")["amount"].sum().asfreq("D").fillna(0)
    model = ARIMA(daily_cashflow, order=(2,1,2))
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

    # Identify peaks in recent cashflow
    recent = daily_cashflow[-30:]
    recent_peaks = recent[(recent.shift(1) < recent) & (recent.shift(-1) < recent)]

    # Annotate peaks in recent cashflow
    for idx, value in recent_peaks.items():
        plt.annotate(
            idx.strftime('%Y-%m-%d'),
            (idx, value),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=8,
            color='blue'
        )

    # Identify peaks in forecast series
    forecast_peaks = forecast_series[(forecast_series.shift(1) < forecast_series) & (forecast_series.shift(-1) < forecast_series)]

    # Annotate peaks in forecast
    for idx, value in forecast_peaks.items():
        plt.annotate(
            idx.strftime('%Y-%m-%d'),
            (idx, value),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=8,
            color='red'
        )

    plt.title("10-Day Cash Flow Forecast")
    plt.xlabel("Date")
    plt.ylabel("Amount (INR)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj
