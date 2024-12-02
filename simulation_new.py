from agentTerritoryOrchestrator import AgentTerritoryOrchestrator
from map import CustomGraph
import random
import numpy as np

# Built by ChatGPT
def main():
    # Set random seed for reproducibility
    seed = 9999  # You can change this value to generate different maps
    random.seed(seed)
    np.random.seed(seed)

    num_agents = 4  # Number of zones and agents
    num_nodes = 40
    hub_node = 1
    max_load = 100
    
    # Available heuristics: 
    # 'manhattan_distance', 'euclidean_distance', 'modified_euclidean_distance', 
    # 'congestion', 'zero_heuristic'
    heuristic_type = 'congestion'  # Change this to use different heuristics

    graph = CustomGraph()
    graph.build_a_map(num_zones=num_agents, num_nodes=num_nodes, seed=seed)

    orchestrator = AgentTerritoryOrchestrator(
        graph=graph,
        num_agents=num_agents,
        hub_node=hub_node,
        max_load=max_load,
        heuristic_type=heuristic_type
    )

    print(f"Initialized {len(orchestrator.agents)} agents with {heuristic_type} heuristic")
    print(f"Using random seed: {seed}")
    orchestrator.run_simulation()


if __name__ == "__main__":
    main()
