from map import CustomGraph
from agent import Agent
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.patches as mpatches
import random
from collections import defaultdict, Counter

# Built by ChatGPT
class AgentTerritoryOrchestrator:
    def __init__(self, graph, num_agents, hub_node=1, max_load=1000, heuristic_type='manhattan_distance'):
        """
        Initialize the orchestrator with the graph, number of agents (zones),
        hub node, and maximum load for agents.
        """
        self.graph = graph
        self.num_agents = num_agents
        self.hub_node = hub_node
        self.max_load = max_load
        self.heuristic_type = heuristic_type
        self.agents = []

        # Define distinct colors for zones
        self.zone_colors = self.generate_zone_colors(num_agents)

        # Assign zones to the graph
        self.assign_zones()

        # Create agents
        self.create_agents()

    def generate_zone_colors(self, num_zones):
        """
        Generate distinct colors for each zone dynamically based on the number of zones.
        Uses the HSV colormap to generate colors with evenly spaced hues.
        """
        cmap = plt.get_cmap('hsv')
        return [cmap(i / num_zones) for i in range(num_zones)]

    def assign_zones(self):
        """
        Assign zones to nodes using KMeans clustering based on node positions,
        ensuring balance based on total edge weights.
        Then assign zones to edges based on their connected nodes.
        """
        # Extract node positions
        node_ids = list(self.graph.G.nodes())
        positions = np.array([self.graph.positions[node] for node in node_ids])

        # Calculate total edge weight per node
        node_weights = {node: 0 for node in node_ids}
        for u, v, data in self.graph.G.edges(data=True):
            node_weights[u] += data['weight']
            node_weights[v] += data['weight']

        # Duplicate node positions based on their total edge weights
        # To prevent excessive duplication, scale weights
        max_dup = 10  # Maximum number of duplicates per node
        scaled_weights = {node: min(int((data / max(node_weights.values())) * max_dup), max_dup)
                          for node, data in node_weights.items()}

        # Create a new list of positions with duplicates
        duplicated_positions = []
        for node, pos in zip(node_ids, positions):
            duplicates = scaled_weights[node]
            for _ in range(duplicates):
                duplicated_positions.append(pos)

        # Handle the case where duplicated_positions might be empty
        if not duplicated_positions:
            duplicated_positions = positions

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=self.num_agents, random_state=42)
        labels = kmeans.fit_predict(duplicated_positions)

        # Assign cluster labels back to original nodes
        node_labels = defaultdict(list)
        idx = 0
        for node in node_ids:
            duplicates = scaled_weights[node]
            node_labels[node].extend(labels[idx:idx + duplicates])
            idx += duplicates

        final_labels = {}
        for node in node_ids:
            if node_labels[node]:
                label = Counter(node_labels[node]).most_common(1)[0][0]
                final_labels[node] = label
            else:
                label = random.randint(0, self.num_agents -1)
                final_labels[node] = label

        # Assign zone_id and zone_color to nodes
        for node, label in final_labels.items():
            self.graph.G.nodes[node]['zone_id'] = label
            self.graph.G.nodes[node]['zone_color'] = self.zone_colors[label]

        # Assign zone_id and zone_color to edges based on the zone of the starting node
        for u, v in self.graph.G.edges():
            zone_id = self.graph.G.nodes[u]['zone_id']
            self.graph.G.edges[u, v]['zone_id'] = zone_id
            self.graph.G.edges[u, v]['zone_color'] = self.zone_colors[zone_id]
            self.graph.G.edges[u, v]['label'] = zone_id +1

    def create_agents(self):
        """
        Create agents corresponding to each zone.
        """
        for agent_id in range(1, self.num_agents + 1):
            new_agent = Agent(
                graph=self.graph,
                start_node=self.hub_node,
                hub_node=self.hub_node,
                agent_id=agent_id,
                max_load=self.max_load,
                heuristic_type=self.heuristic_type
            )
            self.agents.append(new_agent)
            print(f"Created Agent {agent_id}")

    def update_traversing_edges(self):
        """
        Update the graph's traversing_edges based on agents' current traversing edge.
        """
        # Clear previous traversing edges
        self.graph.traversing_edges = set()

        for agent in self.agents:
            if agent.edge_being_crossed:
                self.graph.traversing_edges.add(agent.edge_being_crossed)

    def draw_map_with_zones(self, agents, step_time=0.5):
        """
        Draw the map with zones, agents, and traversing edges.
        """
        self.graph.draw_graph_with_agent(agents, step_time=step_time, zone_colors=self.zone_colors)

        # Additionally, label the zones on the map
        # Calculate centroids of zones to place labels
        zone_centroids = {}
        for node in self.graph.G.nodes():
            zone_id = self.graph.G.nodes[node].get('zone_id', -1)
            if zone_id not in zone_centroids:
                # Initialize as float to allow division later
                zone_centroids[zone_id] = np.array(self.graph.positions[node], dtype=float)
            else:
                zone_centroids[zone_id] += np.array(self.graph.positions[node], dtype=float)

        zone_counts = {}
        for node in self.graph.G.nodes():
            zone_id = self.graph.G.nodes[node].get('zone_id', -1)
            zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1

        # Compute average position for centroids
        for zone_id in zone_centroids:
            zone_centroids[zone_id] /= zone_counts[zone_id]

        # Add zone labels
        for zone_id, centroid in zone_centroids.items():
            if zone_id == -1:
                continue  # Skip if zone_id was not assigned
            plt.text(centroid[0], centroid[1], f'Zone {zone_id +1}',
                     fontsize=12, fontweight='bold', color='black',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    def run_simulation(self):
        """
        Run the simulation loop.
        """
        # Create figure
        fig = plt.figure(figsize=(14, 12))

        # Set up pause state and next step flag
        self.paused = False
        self.next_step = False

        def on_key_press(event):
            if event.key == ' ':  # Space bar
                self.paused = not self.paused
                if self.paused:
                    print("\nSimulation paused. Press space to continue or 'n' for next step.")
                else:
                    print("\nSimulation resumed.")
            elif event.key == 'n' and self.paused:  # 'n' key only works when paused
                self.next_step = True
                print("\nExecuting next step...")

        def on_close(event):
            plt.close('all')
            exit(0)

        # Connect event handlers
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('close_event', on_close)

        # Visualize the initial map with the agents' positions
        self.draw_map_with_zones(self.agents, step_time=0.5)

        # Run the simulation
        step_count = 0
        simulation_active = True

        while simulation_active:
            # Check if simulation is paused
            while self.paused and not self.next_step:
                plt.pause(0.1)  # Keep GUI responsive while paused
                if not plt.get_fignums():  # Check if window was closed
                    return

            # Reset next step flag
            self.next_step = False

            print(f"\n--- Step {step_count} ---")

            # Track if any agent is still active
            any_agent_active = False

            # Update each agent
            for agent in self.agents:
                agent_active = agent.step()
                any_agent_active = any_agent_active or agent_active

            # Update the graph (congestion only, no new targets)
            self.graph.update_map()

            # Update traversing edges based on agents' current actions
            self.update_traversing_edges()

            # Visualize the current state
            self.draw_map_with_zones(self.agents, step_time=0.5)

            # Increment step counter
            step_count += 1

            # Check if simulation should continue
            if not any_agent_active:
                all_at_hub = all(agent.current_node == self.hub_node for agent in self.agents)
                if not all_at_hub:
                    # Continue simulation until all agents reach the hub
                    simulation_active = True
                    for agent in self.agents:
                        if agent.current_node != self.hub_node and agent.state == 'idle':
                            # Plan path back to the hub if the agent is idle
                            agent.plan_path_to_hub()
                            agent.state = 'moving'
                            agent.mission = 'returning'
                else:
                    simulation_active = False

        # Final visualization
        self.draw_map_with_zones(self.agents, step_time=0.5)

        # Print final statistics
        print("\nSimulation Complete!")
        print(f"Total steps taken: {step_count}")
        for agent in self.agents:
            print(f"Agent {agent.agent_id} collected {agent.collected_targets} targets")

        # Calculate and print total targets collected
        total_targets = sum(agent.collected_targets for agent in self.agents)
        print(f"Total targets collected by all agents: {total_targets}")

        plt.show()
