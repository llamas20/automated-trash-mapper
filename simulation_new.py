#Built by ChatGPT
from map import CustomGraph
from agent import Agent
import matplotlib.pyplot as plt

def build_big_map():
    """
    Build a larger map using Delaunay triangulation.
    """
    graph = CustomGraph()
    graph.build_a_map()  # Generates a map with 50 nodes
    return graph

# Exits the figure upon closing the window
def handle_close(evt):
    raise SystemExit("Closed figure, exiting program")

def main():
    # Initialize interactive mode and set up the figure
    plt.ion()
    fig = plt.figure(figsize=(12, 12))  # Initialize the figure once
    fig.canvas.mpl_connect('close_event', handle_close)
    # Build the big map
    graph = build_big_map()

    # Define the hub node (assuming node 1 is the hub)
    hub_node = 1

    # Initialize the agent at the hub node
    agent = Agent(graph=graph, start_node=hub_node, hub_node=hub_node)
    print("Agent initialized")

    # Visualize the initial map with the agent's position
    graph.draw_graph_with_agent([agent], step_time=0.5)

    # Run the simulation
    step_count = 0
    while True:
        print(f"\n--- Step {step_count} ---")
        continue_simulation = agent.step()
        step_count += 1

        # Visualize the map after each step with the agent's current position
        graph.draw_graph_with_agent([agent], step_time=0.5)

        if not continue_simulation:
            break

    # After collecting all targets, ensure the agent is at the hub
    if agent.current_node != hub_node:
        print("\n--- Returning to Hub ---")
        agent.return_to_hub()
        graph.draw_graph_with_agent([agent], step_time=0.5)

    print("\nSimulation Complete.")
    print(f"Total targets collected: {agent.collected_targets}")

    # Finalize the plot display
    plt.ioff()       # Deactivate interactive mode
    plt.show()       # Keep the plot window open until manually closed

if __name__ == "__main__":
    main()
