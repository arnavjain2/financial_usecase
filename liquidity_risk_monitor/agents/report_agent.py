import matplotlib.pyplot as plt
from util.helpers import gemini_simple_chat

def report_agent(risk, reasons, features, gemini_analysis):
    prompt = f"""
You are a senior treasury analyst. Prepare a professional liquidity risk report.

Risk Level: {risk.upper()}
Financial Summary: {features}
Detailed Analysis & solutions: {gemini_analysis}

Structure:
- Executive summary
- Financial position
- Risk analysis (with reasons)
- Key risk factors
- Impact assessment
- Recommendations
- Monitoring
(no markdown or code, just clear human prose)
"""
    try:
        print("Gemini report generation...")
        response_text = gemini_simple_chat(prompt, timeout=60)
        print("Gemini report generation complete.")
        return response_text
    except Exception as e:
        print(f"Gemini report generation error: {e}")
        return (f"Liquidity Risk Report (fallback):\n"
                f"Risk Level: {risk.upper()}\n\n"
                f"Summary: {str(reasons)}\n"
                f"Use manual review for business recommendations.")

def show_forecast(daily_cashflow, forecast_series):
    plt.figure(figsize=(12, 6))
    plt.plot(daily_cashflow[-30:], label="Past 30 Days Cashflow")
    plt.plot(forecast_series, linestyle="--", color="red", label="10-Day Forecast")
    plt.title("Projected 10-Day Cashflow Forecast")
    plt.xlabel("Date")
    plt.ylabel("Amount (INR)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
