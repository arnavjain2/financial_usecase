from agents.supervisor_agent import supervisor_graph, RiskState

if __name__ == "__main__":
    print("Launching Gemini-powered Liquidity Risk Supervisor...")
    initial_state: RiskState = {}
    supervisor_graph.invoke(initial_state)
