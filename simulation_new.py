from agentTerritoryOrchestrator import AgentTerritoryOrchestrator
from map import CustomGraph


def main():
    num_agents = 3  # Number of zones and agents
    num_nodes = 10
    hub_node = 1
    max_load = 100

    graph = CustomGraph()
    graph.build_a_map(num_zones=num_agents, num_nodes=num_nodes)

    orchestrator = AgentTerritoryOrchestrator(
        graph=graph,
        num_agents=num_agents,
        hub_node=hub_node,
        max_load=max_load
    )

    print(f"Initialized {len(orchestrator.agents)} agents")
    orchestrator.run_simulation()


if __name__ == "__main__":
    main()
