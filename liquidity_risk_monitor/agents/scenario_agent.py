import json
from pathlib import Path

def load_scenario_data():
    """Load the monthly cash flow scenario JSON."""
    data_path = Path(__file__).parent.parent / "data" / "monthly_cashflow_scenarios.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def analyze_scenarios(scenarios):
    """Analyze monthly data to detect patterns and risks."""
    report_lines = []
    report_lines.append("### Scenario Agent Liquidity Risk Report\n")
    report_lines.append("**12-Month Historical Cashflow Stress Analysis**\n")
    
    risk_months = [s for s in scenarios if s["risk_event"] == "Detected"]
    
    if not risk_months:
        report_lines.append("✅ No historical liquidity risks detected over the last 12 months.\n")
    else:
        report_lines.append(f"⚠️ Detected {len(risk_months)} months with liquidity risk events.\n")
        report_lines.append("\n| Month | Inflow (INR) | Outflow (INR) | Balance (INR) | Reason |\n| --- | --- | --- | --- | --- |")
        for r in risk_months:
            line = f"| {r['month']} | ₹{r['inflow']:,} | ₹{r['outflow']:,} | ₹{r['balance']:,} | {r['risk_reason']} |"
            report_lines.append(line)
        
        # Summary
        report_lines.append("\n**Summary Analysis:**")
        report_lines.append("* Historical data shows repeated liquidity stress in certain months.")
        report_lines.append("* Key triggers: unexpected large payments, FX losses, delayed receivables.")
        report_lines.append("* Recommend strengthening cash buffer and forecasting processes.")

    return "\n".join(report_lines)
