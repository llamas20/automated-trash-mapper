from map import CustomGraph
from agent import Agent
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.patches as mpatches
import random
from collections import defaultdict, Counter
import networkx as nx

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

            # Update the graph (congestion only, no new targets)
            self.graph.update_map()

            # Check if redistribution is triggered for Territory Malleability
            if self.redistribution_condition({"step_count":step_count}):
                self.redistribute()

            # Update each agent
            for agent in self.agents:
                agent_active = agent.step()
                any_agent_active = any_agent_active or agent_active

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

    def redistribution_condition(self, params):
        """
        This function dictates when to redistribute the graph territories
        """
        freq = 10 # how many steps between each redistribution
        if params["step_count"] % freq == 0 and params["step_count"] != 0:
            return True
        else:
            return False

    def calculate_work(self, edge):
        """
        This function dictates how to calculate the work for each edge for redistribution.
        """
        Target = edge['targets']
        Weight = edge['weight']

        targetPercentage = 0.85
        if edge["congestion_prone"]:
            targetPercentage *= 0.85 # reduce weight of target when congestion prone

        calc_w = Target*targetPercentage + Weight*(1-targetPercentage)

        return calc_w
    
    def calculate_zone_work(self):
        """
        Calulcates the amount of work per zone
        """
        num_zones = self.num_agents
        zone_weights = [0]*num_zones
        for u, v in self.graph.G.edges():
            zone_id = self.graph.G.edges[u, v]['zone_id']
            #print([u, v])
            zone_weights[zone_id] += self.calculate_work(self.graph.G.edges[u, v])
        return zone_weights
    
    def find_zone_edges(self, G, start_node, zone_id=1):
        """
        Function to find edges with a specific zone_id starting from a start_node
        """
        visited = set()  # To track visited nodes
        edges_seen = set()  # To store edges that satisfy the zone_id condition

        # Depth-First Search (DFS) function
        def dfs(node):
            visited.add(node)

            # Traverse each neighbor
            for neighbor in G.neighbors(node):
                # Check if the edge has not been visited and if the edge has zone_id == zone_id
                edge = (node, neighbor) if node < neighbor else (neighbor, node)
                if edge not in edges_seen and G.edges[node, neighbor]['zone_id'] == zone_id:
                    edges_seen.add(edge)  # Add edge to the set of seen edges
                    dfs(neighbor)  # Continue DFS on this neighbor

        # Start DFS from the start_node
        dfs(start_node)

        return edges_seen

    def discontinuity(self, edge):
        zone = edge['zone_id']
        G = self.graph.G
        #print("Discontinuity edge check", edge)

        # get nodes
        nodeA = None
        nodeB = None
        for u, v in G.edges():
            if G.edges[u,v] == edge:
                nodeA = u
                nodeB = v


        # determine which node boarders another zone
        start_node = nodeA
        for u, v in G.edges(nodeA):
            if G.edges[u,v]['zone_id'] != zone:
                start_node = nodeB

        # start edge count from other node
            # count all connected nodes with DFS or BFS
        edges_before = self.find_zone_edges(G, start_node, zone_id=zone)
        count_of_edges_before_remove = len(edges_before)
        
        # change edge zone label temporarily
        edge['zone_id'] = self.num_agents

        # start edge count from same node that did not boarder
            # count all connected nodes with DFS or BFS
        edges_after = self.find_zone_edges(G, start_node, zone_id=zone)
        count_of_edges_after_remove = len(edges_after)

        if count_of_edges_before_remove != count_of_edges_after_remove + 1:
            # discontinuity found
            # print("Discontinuity check before and after edge counts:",count_of_edges_before_remove, count_of_edges_after_remove)
            edge['zone_id'] = zone
            return True

        # revert edge label
        return False

       
    def minimize_zone_work_diff(self, edge1, edge2, zone_work):
        """
        Determine weather to switch edge between zones and then switch them
        """
        # Calculate the work of both edges
        edge1_work = self.calculate_work(edge1)
        edge2_work = self.calculate_work(edge2)
        
        # Get the current zone ids for the edges
        edge1_zone = edge1['zone_id']
        edge2_zone = edge2['zone_id']
        
        # Calculate the current difference in work between the zones
        zone_work_diff_before = zone_work[edge1_zone] - zone_work[edge2_zone]
        
        #print("zone_work_diff before:", zone_work_diff_before)
        
        # Calculate what the new zone work would be if we switch the edges
        zone_work_after_switch1 = (zone_work[edge1_zone] - edge1_work) + edge2_work  # If edge1 goes to edge2's zone
        zone_work_after_switch2 = (zone_work[edge2_zone] - edge2_work) + edge1_work  # If edge2 goes to edge1's zone
        
        # Calculate the new differences in work if the switch happens
        zone_work_diff_after1 = zone_work_after_switch1 - zone_work[edge2_zone]
        zone_work_diff_after2 = zone_work_after_switch2 - zone_work[edge1_zone]
        
        # We want to minimize the absolute difference in work between the two zones
        if abs(zone_work_diff_after1) < abs(zone_work_diff_before):
            # Determine if switch will cause discontinuity of zone
            if self.discontinuity(edge1):
                # if discontinuity by switching edge label, then don't switch
                return
            # Switching edge1 to edge2's zone reduces the difference
            print("Added edge from zone", edge1['label'], "to zone", edge2['label'] )
            zone_work[edge1_zone] -= edge1_work
            zone_work[edge2_zone] += edge1_work
            edge1['zone_id'] = edge2_zone
            edge1['label'] = edge2['label']
            #print("Switched edge1 to edge2's zone.")
        elif abs(zone_work_diff_after2) < abs(zone_work_diff_before):
            # Determine if switch will cause discontinuity of zone
            if self.discontinuity(edge1):
                # if discontinuity by switching edge label, then don't switch
                return
            # Switching edge2 to edge1's zone reduces the difference
            print("Added edge from zone", edge2['label'], "to zone", edge1['label'] )
            zone_work[edge2_zone] -= edge2_work
            zone_work[edge1_zone] += edge2_work
            edge2['zone_id'] = edge1_zone
            edge2['label'] = edge1['label']
            #print("Switched edge2 to edge1's zone.")
        
        # Print the final difference after the potential switch
        #zone_work_diff_after = zone_work[edge1_zone] - zone_work[edge2_zone]
        #print("zone_work_diff after:", zone_work_diff_after)

        return
    

    def redistribute(self):
        """
        Redistributes the graph labels to balance edge weights and target counts. 
        """
        print("Debugging ReDist:")

        num_zones = self.num_agents
        num_passes = 1 # this is number of times redistribution is run
        
        # Calculate total amount of "work" per zone
        zone_work = self.calculate_zone_work()
        
        print(zone_work)
        print("Zone work variance BEFORE redistribution:",np.var(zone_work))

        # get target work for each zone   
        target_work = sum(zone_work)/num_zones


        # trade edges between zones to minimize work differences
        for i in range(num_passes):
            for u, v in self.graph.G.edges():
                for w, m in self.graph.G.edges():
                    if u == w or v == m: # same edge, skip
                        continue
                    elif w == v or m == v or u == m or u == w: # edges share a node / are touching
                        # minimize work diff if edges in different zones
                        if self.graph.G.edges[u,v]["label"] != self.graph.G.edges[w,m]["label"]:
                            self.minimize_zone_work_diff(self.graph.G.edges[u,v], self.graph.G.edges[w,m], zone_work)

            print(zone_work)
            print(f"Zone work variance AFTER redistribution iteration {i} of {num_passes}:",np.var(zone_work))

        print("End ReDist")
        return
