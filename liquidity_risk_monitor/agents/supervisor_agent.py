from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Any, List
from util.helpers import load_data, forecast_with_arima, to_native
from agents.scenario_agent import scenario_agent
from agents.rating_agent import rating_agent
from agents.validator_agent import validator_agent
from agents.report_agent import report_agent, show_forecast
import datetime
import os
import json

class RiskState(TypedDict):
    data: dict
    daily_cashflow: Any
    forecast_series: Any
    logs_summary: str
    features: dict
    scenario_reasons: List[str]
    risk: str
    reasons: List[str]
    gemini_analysis: dict
    report: str
    refined_report: str

def load_and_forecast(state: RiskState) -> RiskState:
    data = load_data()
    daily_cashflow, forecast_series = forecast_with_arima(data["transactions"])
    state["data"] = data
    state["daily_cashflow"] = daily_cashflow
    state["forecast_series"] = forecast_series
    return state

def run_scenario(state: RiskState) -> RiskState:
    logs_summary, features, scenario_reasons = scenario_agent(state["data"])
    state["logs_summary"] = logs_summary
    state["features"] = features
    state["scenario_reasons"] = scenario_reasons
    return state

def rate_risk(state: RiskState) -> RiskState:
    risk, reasons, gemini_analysis = rating_agent(state["features"], state["scenario_reasons"])
    state["risk"] = risk
    state["reasons"] = reasons
    state["gemini_analysis"] = gemini_analysis
    return state

def create_report(state: RiskState) -> RiskState:
    state["report"] = report_agent(state["risk"], state["reasons"], state["features"], state["gemini_analysis"])
    return state

def refine_report(state: RiskState) -> RiskState:
    state["refined_report"] = validator_agent(state["report"])
    return state

def save_log(state: RiskState) -> RiskState:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": timestamp,
        "features": to_native(state["features"]),
        "risk_predicted": state["risk"],
        "reasons": state["reasons"],
        "scenarios_analyzed": state["scenario_reasons"],
        "gemini_analysis": to_native(state["gemini_analysis"]),
        "result": state["refined_report"],
        "risk_detected": state["risk"],
    }
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/liquidity_risk_log_{timestamp}.json", "w") as f:
        json.dump(log, f, indent=2)
    return state

def maybe_plot(state: RiskState) -> RiskState:
    # No interactive plotting here; frontend will handle chart display
    return state

def print_result(state: RiskState) -> RiskState:
    # No terminal printout; only return state for frontend to display
    return state

graph_builder = StateGraph(RiskState)
graph_builder.add_node("load_and_forecast", load_and_forecast)
graph_builder.add_node("scenario", run_scenario)
graph_builder.add_node("rate", rate_risk)
graph_builder.add_node("create_report", create_report)
graph_builder.add_node("refine_report", refine_report)
graph_builder.add_node("save_log", save_log)
graph_builder.add_node("maybe_plot", maybe_plot)
graph_builder.add_node("print_result", print_result)

graph_builder.add_edge(START, "load_and_forecast")
graph_builder.add_edge("load_and_forecast", "scenario")
graph_builder.add_edge("scenario", "rate")
graph_builder.add_edge("rate", "create_report")
graph_builder.add_edge("create_report", "refine_report")
graph_builder.add_edge("refine_report", "save_log")
graph_builder.add_edge("save_log", "maybe_plot")
graph_builder.add_edge("maybe_plot", "print_result")
graph_builder.add_edge("print_result", END)

supervisor_graph = graph_builder.compile()
