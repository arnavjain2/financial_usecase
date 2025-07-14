from util.helpers import load_data, forecast_with_arima, plot_forecast
from agents.rating_agent import generate_rating
from agents.report_agent import generate_detailed_report, save_feedback

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    daily_cashflow, forecast_series = forecast_with_arima(data["transactions"])

    print("Running Rating Agent...")
    risk_rating, rating_response = generate_rating(data, forecast_series)
    print(f"Rating Agent says: {risk_rating.upper()}")
    print(" Gemini Output:", rating_response)

    if risk_rating in ["medium", "high"]:
        print("⚠️ Risk is present. Generating full report...\n")
        report = generate_detailed_report(data, forecast_series)
        print(report)
        plot_forecast(daily_cashflow, forecast_series)
        save_feedback(report, forecast_series)
    else:
        print(" Supervisor Agent: Risk is LOW. No further action needed.")
