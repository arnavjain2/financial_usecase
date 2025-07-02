# Full Agentic Liquidity Risk Management System with Groq LLM and ARIMA Forecasting using CSV Input

import os
import datetime
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from statsmodels.tsa.arima.model import ARIMA

# ========== Load GROQ API ==========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ========== Load Financial Data from CSV Files ==========
def load_data():
    base_path = ""
    data = {
        "bank_data": pd.read_csv(base_path + "bank_data.csv").to_dict(orient="records"),
        "ap_ar_data": pd.read_csv(base_path + "ap_ar_data.csv").to_dict(orient="records"),
        "fx_data": pd.read_csv(base_path + "fx_data.csv").to_dict(orient="records"),
        "cashflow_series": calculate_cashflow_series(),
        "past_logs": load_logs()
    }
    return data

# ========== Calculate Net Cash Flow Series from Unified Transactions ==========
def calculate_cashflow_series():
    df = pd.read_csv("cash_transactions.csv")  # Expected columns: from, to, amount, date, time, description
    df["date"] = pd.to_datetime(df["date"])
    cashflow_series = df.groupby("date")["amount"].sum()
    return cashflow_series

# Load past feedback logs

def load_logs():
    logs = []
    log_path = "logs"
    if os.path.exists(log_path):
        for file in os.listdir(log_path):
            if file.endswith(".json"):
                with open(os.path.join(log_path, file), "r") as f:
                    try:
                        logs.append(json.load(f))
                    except Exception as e:
                        continue
    return logs

# ========== Run ARIMA Forecast ==========
def run_arima(series):
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=3)
    forecast_list = forecast.round(2).tolist()
    return forecast_list

# ========== Format Prompt for Groq ==========
def format_for_groq(data, arima_forecast):
    recent_feedback = "\n\nPast risk decisions:\n" + "\n".join(
        [f"[{log['timestamp']}] Risk: {log['risk_detected']}, Notes: {log.get('user_feedback', '')}" for log in data.get("past_logs", [])[-3:]]
    ) if data.get("past_logs") else ""

    prompt = f"""
You are an expert treasury analyst.

IMPORTANT:
- 'Party_Type: Customer' means this invoice is money **we are expecting to receive**.
- 'Party_Type: Vendor' means this invoice is money **we must pay**.
- All amounts are in INR.

Here is the raw financial data:
Bank Accounts: {data['bank_data']}
Invoices (AP/AR): {data['ap_ar_data']}


Also, based on the net cashflow calculated from real inflows/outflows, the ARIMA model predicts:
Next 3 months cash flow forecast: {arima_forecast}
{recent_feedback}

Task: should take ARIMA into account but not completely rely on it as it doesn't account for upcoming receivables and payables.
Analyze this data and tell:
Give a rating (high, medium or low) for risk
1. Is there any liquidity risk?
2. Think of 2-3 realistic average-case risk scenarios (e.g., delayed payment, emergency expense).
3. Modify cash inflow/outflow assumptions based on those.
4. Forecast daily cash position for the next 10 days.
5. If balance goes negative on any day, identify the date and the detailed reason why and what caused it .
6. Generate a report on why this is happening.
7. Be detailed and precise.
"""
    return prompt

# ========== Run Groq Analysis ==========
def run_groq_analysis(prompt):
    llm = ChatGroq(temperature=0.3, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    response = llm.invoke(prompt)
    return response

# ========== Format Output ==========
def format_output(result_text):
    print("\n" + "=" * 60)
    print("🧾 Liquidity Risk Management Report".center(60))
    print("=" * 60)
    lines = result_text.strip().split("\n")
    for line in lines:
        if any(kw in line.lower() for kw in ["liquidity risk", "recommendation", "driver", "summary"]):
            print(f"\n🔹 {line.strip()}")
        else:
            print(f"   {line.strip()}")
    print("\n" + "=" * 60)

# ========== Save Feedback ==========
def save_feedback(data, result_text):
    data["cashflow_series"] = data["cashflow_series"].rename_axis("date").reset_index()
    data["cashflow_series"]["date"] = data["cashflow_series"]["date"].dt.strftime("%Y-%m-%d")
    data["cashflow_series"] = data["cashflow_series"].to_dict(orient="records")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": timestamp,
        "input_data": data,
        "output": result_text,
        "risk_detected": None,
        "user_feedback": None,
        "notes": ""
    }

    risk_input = input("Was a real risk present? (yes/no): ").strip().lower()
    if risk_input in ["yes", "no"]:
        log["risk_detected"] = risk_input

    feedback_input = input("Optional: Any feedback or notes? (press enter to skip): ").strip()
    if feedback_input:
        log["user_feedback"] = feedback_input

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/liquidity_risk_log_{timestamp}.json", "w") as f:
        json.dump(log, f, indent=2)

# ========== Main ==========
if __name__ == "__main__":
    data = load_data()
    arima_forecast = run_arima(data["cashflow_series"])
    prompt = format_for_groq(data, arima_forecast)
    result = run_groq_analysis(prompt)

    format_output(result.content)
    save_feedback(data, result.content)