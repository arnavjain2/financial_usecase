from util.helpers import gemini_simple_chat
import json

def rating_agent(features, scenario_reasons):
    prompt = f"""
You are a treasury risk expert. Assess the company's liquidity risk using the following data and scenarios.

Financial Data:
- Cash Balance: ₹{features['cash_balance']:.2f}
- Net Cashflow (12m): ₹{features['net_cashflow']:.2f}
- Inflows: ₹{features['total_inflows']:.2f}
- Outflows: ₹{features['total_outflows']:.2f}
- Expected Outflow: ₹{features['expected_outflow']:.2f}

Identified Scenarios:
{chr(10).join([f"- {scenario}" for scenario in scenario_reasons])}

TASK:
1. Assign a risk: HIGH, MEDIUM, or LOW.
2. List all reasons, key risk factors, and concrete risk solutions.

Reply ONLY as valid JSON:
{{
    "risk_rating": "HIGH/MEDIUM/LOW",
    "primary_reasons": ["reason1", "reason2"],
    "risk_factors": ["factor1", "factor2"],
    "recommended_solutions": ["solution1", "solution2"],
    "confidence_level": "High/Medium/Low"
}}
"""
    try:
        print("Gemini rating analysis...")
        response_text = gemini_simple_chat(prompt, timeout=60)
        print("Gemini rating analysis complete.")
        try:
            idx1 = response_text.find("{")
            idx2 = response_text.rfind("}") + 1
            rating_analysis = json.loads(response_text[idx1:idx2])
            risk = rating_analysis.get("risk_rating", "MEDIUM").lower()
            reasons = rating_analysis.get("primary_reasons", []) + rating_analysis.get("risk_factors", [])
            return risk, reasons, rating_analysis
        except Exception as e:
            print(f"Rating JSON parse error: {e}")
            return "medium", [response_text[:300]], {}
    except Exception as e:
        print(f"Gemini rating API error: {e}")
        return "medium", ["Gemini API error"], {}
