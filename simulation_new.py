from map import CustomGraph
from agent import Agent
import matplotlib.pyplot as plt


def build_big_map(num_zones):
    """
    Build a larger map using Delaunay triangulation.
    """
    graph = CustomGraph()
    graph.build_a_map(num_zones=num_zones)  # Generates a map with 50 nodes and labeled edges
    return graph


def create_agents(graph, num_agents, start_node, hub_node):
    """
    Create a list of agents with different IDs.
    """
    agents = []
    for agent_id in range(1, num_agents + 1):
        new_agent = Agent(graph=graph, start_node=start_node, hub_node=hub_node, agent_id=agent_id)
        agents.append(new_agent)
        print(f"Created Agent {agent_id}")
    return agents


def main():
    # Set number of agents/zones
    num_agents = 5  # Change this number to use more agents

    # Build the big map with specified number of zones
    graph = build_big_map(num_agents)

    # Create figure
    fig = plt.figure(figsize=(12, 12))

    # Set up close event handler
    def on_close(event):
        plt.close('all')
        exit(0)

    fig.canvas.mpl_connect('close_event', on_close)

    # Define the hub node (assuming node 1 is the hub)
    hub_node = 1

    # Initialize agents
    agents = create_agents(graph, num_agents=num_agents, start_node=hub_node, hub_node=hub_node)
    print(f"Initialized {len(agents)} agents")

    # Visualize the initial map with the agents' positions
    graph.draw_graph_with_agent(agents, step_time=0.5)

    # Run the simulation
    step_count = 0
    simulation_active = True

    while simulation_active:
        print(f"\n--- Step {step_count} ---")

        # Track if any agent is still active
        any_agent_active = False

        # Update each agent
        for agent in agents:
            agent_active = agent.step()
            any_agent_active = any_agent_active or agent_active

        # Update the graph (congestion only, no new targets)
        graph.update_map()

        # Visualize the current state
        graph.draw_graph_with_agent(agents, step_time=0.5)

        # Increment step counter
        step_count += 1

        # Check if simulation should continue
        if not any_agent_active:
            simulation_active = False

    # After all agents are done, ensure they return to hub
    print("\n--- Final Return to Hub ---")
    for agent in agents:
        if agent.current_node != hub_node:
            agent.return_to_hub()

    # Final visualization
    graph.draw_graph_with_agent(agents, step_time=0.5)

    # Print final statistics
    print("\nSimulation Complete!")
    print(f"Total steps taken: {step_count}")
    for agent in agents:
        print(f"Agent {agent.agent_id} collected {agent.collected_targets} targets")

    # Calculate and print total targets collected
    total_targets = sum(agent.collected_targets for agent in agents)
    print(f"Total targets collected by all agents: {total_targets}")

    plt.show()


if __name__ == "__main__":
    main()