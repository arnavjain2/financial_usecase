from util.helpers import gemini_simple_chat

def validator_agent(report_text):
    prompt = f"""
You are a senior risk report reviewer and editor.

TASK:
Review the following liquidity risk report for factual inaccuracies, logic errors, unclear recommendations, poor structure, or grammar. Fix any found: clarify, correct, improve flow and professionalism. If the report is excellent, simply return it—no commentary.

REPORT TO REVIEW:
{report_text}
"""
    try:
        print("Gemini report refinement...")
        refined = gemini_simple_chat(prompt, timeout=60)
        print("Gemini refinement complete.")
        return refined
    except Exception as e:
        print(f"Gemini validator/refiner error: {e}")
        return report_text
