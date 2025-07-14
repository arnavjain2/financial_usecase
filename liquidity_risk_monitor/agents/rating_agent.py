import google.generativeai as genai
import re
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_risk_rating(text):
    match = re.search(r"(liquidity risk.*?rating.*?:\s*)(high|medium|low)", text, re.IGNORECASE)
    return match.group(2).lower() if match else "unknown"

def generate_rating(data, forecast_series):
    forecast_txt = "\n".join([f"{date.date()}: ₹{value:,.2f}" for date, value in forecast_series.items()])
    prompt = f"""
Here is a 10-day INR cashflow forecast:
{forecast_txt}
dont try to generate risk forcefully .
Please answer only the following:
1. Is there any liquidity risk?
2. If yes, give a risk rating (high, medium, low) on cashflow only.

Answer in one line. Example: "Liquidity risk rating: Medium"
"""
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    rating = extract_risk_rating(response.text)
    return rating, response.text
