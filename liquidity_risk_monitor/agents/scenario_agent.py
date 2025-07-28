import pandas as pd
import json
from util.helpers import gemini_simple_chat

def extract_features(data):
    df = pd.DataFrame(data["transactions"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["currency"] == "INR"]
    money_in = df[df["amount"] > 0]["amount"].sum()
    money_out = -df[df["amount"] < 0]["amount"].sum()
    closing_balance = pd.DataFrame(data["bank_data"])["Available_Balance"].sum()
    expected_outflow = pd.DataFrame(data["fixed_payments"])["amount"].sum()
    net_change = money_in - money_out
    return {
        "total_inflows": money_in,
        "total_outflows": money_out,
        "net_cashflow": net_change,
        "expected_outflow": expected_outflow,
        "cash_balance": closing_balance
    }

def scenario_agent(data):
    features = extract_features(data)
    bank_data_summary = f"Bank Accounts: {data['bank_data'][:2]}"
    transactions_summary = f"Recent Transactions: {data['transactions'][-10:]}"
    fixed_payments_summary = f"Fixed Payments: {data['fixed_payments'][:10]}"
    logs_summary = f"Past Analysis Logs: {data['logs'][-3:]}"
    prompt = f"""
You are a senior treasury risk analyst. Analyze the following financial data and generate comprehensive liquidity risk scenarios.

FINANCIAL DATA:
- Cash Balance: ₹{features['cash_balance']:.2f}
- Net Cashflow (12 months): ₹{features['net_cashflow']:.2f}
- Total Inflows: ₹{features['total_inflows']:.2f}
- Total Outflows: ₹{features['total_outflows']:.2f}
- Expected Future Outflows: ₹{features['expected_outflow']:.2f}
{bank_data_summary}
{transactions_summary}
{fixed_payments_summary}
{logs_summary}

1. Analyze for concerning trends.
2. Produce 5-7 detailed, plausible risk scenarios (historical, novel, market etc)—each with a name, description, likely impact, and probability.
Respond ONLY as valid JSON:
{{
"historical_analysis": "Summary",
"scenarios": [{{"scenario_name": "", "description": "", "potential_impact": "", "likelihood": ""}}]
}}
"""
    try:
        print("Gemini scenario analysis...")
        response_text = gemini_simple_chat(prompt, timeout=60)
        print("Gemini scenario analysis complete.")
        try:
            idx1 = response_text.find("{")
            idx2 = response_text.rfind("}") + 1
            scenario_analysis = json.loads(response_text[idx1:idx2])
        except Exception as e:
            print(f"JSON parse error: {e}")
            scenario_analysis = {
                "historical_analysis": response_text,
                "scenarios": [{
                    "scenario_name": "AI Generated Scenario",
                    "description": response_text,
                    "potential_impact": "To be assessed",
                    "likelihood": "Medium"
                }]
            }
        scenario_reasons = []
        for scenario in scenario_analysis.get("scenarios", []):
            scenario_reasons.append(f"{scenario['scenario_name']}: {scenario['description']}")
        return scenario_analysis.get("historical_analysis", ""), features, scenario_reasons
    except Exception as e:
        print(f"Gemini scenario error: {e}")
        fallback_scenarios = [
            "Cashflow pattern suggests liquidity constraints.",
            "Upcoming fixed payments may strain reserves.",
            "Seasonal variation may pose risks."
        ]
        return "Fallback analysis due to API error", features, fallback_scenarios
