import google.generativeai as genai
from dotenv import load_dotenv
import os
import datetime
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_detailed_report(data, forecast_series):
    forecast_txt = "\n".join([f"{date.date()}: ₹{value:,.2f}" for date, value in forecast_series.items()])
    recent_feedback = "\n\nPast risk decisions:\n" + "\n".join(
        [f"[{log['timestamp']}] Risk: {log['risk_detected']}, Notes: {log.get('user_feedback', '')}" 
         for log in data.get("past_logs", [])[-3:]]
    ) if data.get("past_logs") else ""

    prompt = f"""
You are a treasury risk analyst.

Here's INR-based forecasted cashflow:
{forecast_txt}

Historical Data:
Bank Accounts: {data['bank_data']}
Fixed payments: {data['fixed_payments']}
Transactions: {data['transactions'].to_dict(orient="records")}
{recent_feedback}

Give:
1. 2-3 risk scenarios.
2. Adjust cashflows based on those.
3. Forecast position after those scenarios.
4. If cash goes negative, give reason and day.
5. Final risk report.
"""
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text

def save_feedback(result_text, forecast_series):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": timestamp,
        "input_data": {
             "forecast": {str(k): float(v) for k, v in forecast_series.items()},  
                       },
        "output": result_text,
        "risk_detected": None,
        "user_feedback": None
    }

    print("\nWas the analysis correct? (yes/no):")
    risk_input = input().strip().lower()
    if risk_input in ["yes", "no"]:
        log["risk_detected"] = risk_input

    print("Any feedback? (press enter to skip):")
    feedback = input().strip()
    if feedback:
        log["user_feedback"] = feedback

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/liquidity_risk_log_{timestamp}.json", "w") as f:
        json.dump(log, f, indent=2)
