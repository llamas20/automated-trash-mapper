import networkx as nx
from map import CustomGraph
import math

#Built by ChatGPT
class Agent:
    def __init__(self, graph, start_node, hub_node, agent_id,max_load, heuristic_type='manhattan_distance'):
        """
        Initialize agent on the graph at a starting node.

        :param graph: Instance of CustomGraph
        :param start_node: Node where the agent starts
        :param hub_node: Node designated as the hub
        :param agent_id: ID that determines which edges the agent can target
        """
        self.graph = graph  # The CustomGraph instance
        self.current_node = start_node  # The current node the agent is at
        self.hub_node = hub_node  # The hub node
        self.agent_id = agent_id  # Agent's ID for targeting specific edges
        self.max_load = max_load
        self.heuristic_type = heuristic_type

        if start_node not in graph.G:
            raise ValueError(f"Node {start_node} does not exist in the graph")
        if hub_node not in graph.G:
            raise ValueError(f"Hub node {hub_node} does not exist in the graph")
        self.collected_targets = 0  # Total targets collected

        # Agent's state management
        self.state = 'idle'  # Possible states: 'idle', 'moving'
        self.mission = None  # 'collecting', 'returning'
        self.load = 0  # Current load
        self.edge_being_crossed = None  # (node1, node2)
        self.time_remaining_on_edge = 0  # Time remaining to traverse the edge
        self.planned_path = []  # List of nodes in planned path
        self.next_node_index = 0  # Index of next node in planned path

    def heuristic(self, node1, node2):
        """Heuristic function for A* (minimum edge weight * manhattan distance)."""
        x1, y1 = self.graph.positions[node1]
        x2, y2 = self.graph.positions[node2]
        # Use Manhattan distance
        manhattan_distance = abs(x1 - x2) + abs(y1 - y2)
        # Find the minimum edge weight in the graph
        min_edge_weight = min(data['weight'] for _, _, data in self.graph.G.edges(data=True))
        # Use minimum edge weight as unit distance cost
        return (manhattan_distance * min_edge_weight) / 10  # Divide by 10 to further ensure no overestimation

    def find_path(self, start, goal):
        """Find path from start to goal using A*."""
        try:
            path = nx.astar_path(self.graph.G, start, goal, heuristic=self.heuristic, weight='weight')
            return path
        except nx.NetworkXNoPath:
            print(f"No path found from {start} to {goal}")
            return None

    def find_nearest_target_edge(self):
        """
        Find the nearest edge with targets from the current node that matches the agent's ID.
        Returns a tuple (edge, path_to_start_of_edge).
        """
        min_distance = float('inf')
        nearest_edge = None
        path_to_edge_start = []

        for edge in self.graph.G.edges(data=True):
            node1, node2, data = edge
            # Check if edge has targets and matches agent's ID
            if data['targets'] > 0 and data['label'] == self.agent_id:
                # Calculate distance to node1
                path1 = self.find_path(self.current_node, node1)
                if path1:
                    distance1 = self.calculate_path_cost(path1)
                    if distance1 < min_distance:
                        min_distance = distance1
                        nearest_edge = (node1, node2)
                        path_to_edge_start = path1

                # Calculate distance to node2
                path2 = self.find_path(self.current_node, node2)
                if path2:
                    distance2 = self.calculate_path_cost(path2)
                    if distance2 < min_distance:
                        min_distance = distance2
                        nearest_edge = (node2, node1)  # Reverse direction
                        path_to_edge_start = path2

        return nearest_edge, path_to_edge_start

    def calculate_path_cost(self, path):
        """Calculate the total weight of the given path."""
        cost = 0
        for i in range(len(path)-1):
            cost += self.graph.G[path[i]][path[i+1]]['weight']
        return cost

    def step(self):
        """Perform one step of the agent's action."""
        if self.state == 'moving':
            # If at hub and mission is returning, reset state
            if self.current_node == self.hub_node and self.mission == 'returning':
                self.state = 'idle'
                self.mission = None
                self.load = 0
                print(f"Agent {self.agent_id} arrived at hub and reset load to zero.")
                return True

            # Remaining movement logic stays unchanged
            self.time_remaining_on_edge -= 1
            if self.time_remaining_on_edge > 0:
                node1, node2 = self.edge_being_crossed
                print(f"Agent {self.agent_id} is moving along edge from {node1} to {node2}, time remaining: {self.time_remaining_on_edge}")
                return True
            else:
                # Arrived at the next node
                self.current_node = self.planned_path[self.next_node_index]
                print(f"Agent {self.agent_id} is now at node {self.current_node}")
                self.next_node_index += 1

                # Check if we have reached the end of the planned path
                if self.next_node_index < len(self.planned_path):
                    # Start moving along the next edge
                    node1 = self.current_node
                    node2 = self.planned_path[self.next_node_index]
                    self.edge_being_crossed = (node1, node2)
                    self.time_remaining_on_edge = self.graph.G[node1][node2]['weight']
                    print(f"Agent {self.agent_id} traversing edge from {node1} to {node2}")
                else:
                    # Reached the end of the path
                    if self.mission == 'collecting':
                        # If we just traversed the edge with targets, collect the targets
                        edge_data = self.graph.G[self.planned_path[-2]][self.planned_path[-1]]
                        node1 = self.planned_path[-2]
                        node2 = self.planned_path[-1]
                        space_remaining = self.max_load - self.load
                        if edge_data['targets'] > 0 and space_remaining > 0:
                            targets_available = edge_data['targets']
                            targets_to_collect = min(targets_available, space_remaining)
                            self.collected_targets += targets_to_collect
                            self.load += targets_to_collect
                            edge_data['targets'] -= targets_to_collect
                            print(
                                f"Agent {self.agent_id} collected {targets_to_collect} targets on edge ({node1}, {node2}). Total collected: {self.collected_targets}. Current load: {self.load}/{self.max_load}")
                            # Set the edge's targets to zero
                            self.graph.update_edge_targets(node1, node2, edge_data['targets'])
                        else:
                            print(f"Agent {self.agent_id} reached edge ({node1}, {node2}) but no targets to collect.")

                        # Decide next action
                        if self.load >= self.max_load:
                            # Load capacity reached, return to hub
                            self.plan_path_to_hub()
                            self.state = 'moving'
                            self.mission = 'returning'
                        else:
                            # Continue collecting targets
                            self.state = 'idle'

                    elif self.mission == 'returning':
                        # Arrived at hub
                        if self.current_node == self.hub_node:
                            self.load = 0
                            print(f"Agent {self.agent_id} arrived at hub and reset load to zero.")
                            self.state = 'idle'
                        else:
                            # Should not happen, but in case
                            self.plan_path_to_hub()
                            self.state = 'moving'
                            self.mission = 'returning'
                    else:
                        # Should not happen
                        self.state = 'idle'
        elif self.state == 'idle':
            print(f"Agent {self.agent_id} is idle at node {self.current_node}.")
            if self.load >= self.max_load:
                # Load capacity reached, return to hub
                self.plan_path_to_hub()
                self.state = 'moving'
                self.mission = 'returning'
            elif self.all_targets_collected():
                # All targets collected
                if self.current_node != self.hub_node:
                    # Return to hub
                    self.plan_path_to_hub()
                    self.state = 'moving'
                    self.mission = 'returning'
                else:
                    # At hub, simulation ends
                    print(f"Agent {self.agent_id} at hub with all targets collected.")
                    return False  # End simulation
            else:
                # Find next target edge
                self.plan_path_to_nearest_target_edge()
                if self.planned_path:
                    self.state = 'moving'
                    self.mission = 'collecting'
                    node1 = self.current_node
                    node2 = self.planned_path[self.next_node_index]
                    print(f"Agent {self.agent_id} traversing edge from {node1} to {node2}")
                else:
                    # No path to any target edge
                    print(f"Agent {self.agent_id}: No path to any target edge.")
                    if self.current_node != self.hub_node:
                        # Return to hub
                        self.plan_path_to_hub()
                        self.state = 'moving'
                        self.mission = 'returning'
                    else:
                        # At hub, simulation ends
                        print(f"Agent {self.agent_id} at hub with no targets remaining.")
                        return False  # End simulation
        return True  # Continue simulation

    def traverse_edge(self, node1, node2):
        """Traverse the edge from node1 to node2 to collect targets."""
        print(f"Agent {self.agent_id} traversing edge from {node1} to {node2}")
        edge_data = self.graph.G[node1][node2]
        targets_collected = edge_data['targets']
        self.collected_targets += targets_collected
        print(
            f"Agent {self.agent_id} collected {targets_collected} targets on edge ({node1}, {node2}). Total collected: {self.collected_targets}")
        # Set the edge's targets to zero as they are collected
        self.graph.update_edge_targets(node1, node2, 0)
        # Update the agent's current node to node2
        self.current_node = node2
        print(f"Agent {self.agent_id} is now at node {self.current_node}")

    def all_targets_collected(self):
        """Check if all targets on edges assigned to this agent are collected."""
        for edge in self.graph.G.edges(data=True):
            if edge[2]['targets'] > 0 and edge[2]['label'] == self.agent_id:
                return False
        return True
    """
    def return_to_hub(self):
        #Find and traverse the path back to the hub node.
        if self.current_node == self.hub_node:
            print(f"Agent {self.agent_id} is already at the hub.")
            return

        path = self.find_path(self.current_node, self.hub_node)
        if path:
            print(f"Agent {self.agent_id} returning to hub via path: {path}")
            for node in path[1:]:
                self.current_node = node
                print(f"Agent {self.agent_id} moved to node {self.current_node}")
        else:
            print(f"Agent {self.agent_id} could not find a path back to the hub.")
    """
    def plan_path_to_hub(self):
        """Plan a path to the hub node and start moving."""
        path = self.find_path(self.current_node, self.hub_node)
        if path and len(path) > 1:  # Ensure path exists and has at least 2 nodes
            self.planned_path = path
            self.next_node_index = 1  # Start from next node in path
            node1 = self.current_node
            node2 = self.planned_path[self.next_node_index]
            self.edge_being_crossed = (node1, node2)
            self.time_remaining_on_edge = self.graph.G[node1][node2]['weight']
            print(f"Agent {self.agent_id} planning path to hub: {self.planned_path}")
        elif self.current_node == self.hub_node:
            # If already at hub, reset state
            self.state = 'idle'
            self.mission = None
            self.planned_path = []
            self.next_node_index = 0
            print(f"Agent {self.agent_id} already at hub.")
        else:
            print(f"Agent {self.agent_id} could not find a path back to the hub.")
            # Reset state to avoid getting stuck
            self.state = 'idle'
            self.mission = None

    def plan_path_to_nearest_target_edge(self):
        """Plan a path to the nearest target edge and start moving."""
        nearest_edge, path_to_edge_start = self.find_nearest_target_edge()
        if nearest_edge and path_to_edge_start:
            edge_start, edge_end = nearest_edge
            # Plan path: path to edge start, then traverse the edge
            self.planned_path = path_to_edge_start
            if self.planned_path[-1] != edge_start:
                # Should not happen
                self.planned_path.append(edge_start)
            # Add the edge traversal to the planned path
            self.planned_path.append(edge_end)
            self.next_node_index = 1  # Start from next node in path
            node1 = self.current_node
            node2 = self.planned_path[self.next_node_index]
            self.edge_being_crossed = (node1, node2)
            self.time_remaining_on_edge = self.graph.G[node1][node2]['weight']
            print(f"Agent {self.agent_id} planning path to target edge {nearest_edge}: {self.planned_path}")
        else:
            # No target edges found
            self.planned_path = None

    def manhattan_distance(self, node1, node2):
        """Original Manhattan distance heuristic"""
        x1, y1 = self.graph.positions[node1]
        x2, y2 = self.graph.positions[node2]
        manhattan_distance = abs(x1 - x2) + abs(y1 - y2)
        min_edge_weight = min(data['weight'] for _, _, data in self.graph.G.edges(data=True))
        return (manhattan_distance * min_edge_weight) / 10

    def euclidean_distance(self, node1, node2):
        """Euclidean distance heuristic"""
        x1, y1 = self.graph.positions[node1]
        x2, y2 = self.graph.positions[node2]
        # Calculate Euclidean distance
        distance = math.hypot(x1 - x2, y1 - y2)
        # Find minimum edge weight
        min_edge_weight = min(data['weight'] for _, _, data in self.graph.G.edges(data=True))
        # Further normalize the distance to ensure it doesn't exceed actual cost
        return (distance * min_edge_weight) / 25  # Increase denominator to further reduce estimate

    def modified_euclidean_distance(self, node1, node2):
        """Modified Euclidean distance with minimum edge weight"""
        x1, y1 = self.graph.positions[node1]
        x2, y2 = self.graph.positions[node2]
        distance = math.hypot(x1 - x2, y1 - y2)
        min_edge_weight = min(data['weight'] for _, _, data in self.graph.G.edges(data=True))
        # Normalize the distance to ensure it doesn't exceed actual cost
        return (distance * min_edge_weight) / 15  # Use appropriate scaling factor

    def calculate_congestion_penalty(self, node1, node2):
        """Calculate congestion penalty between two nodes"""
        congestion_sum = 0
        edge_count = 0
        for u, v, data in self.graph.G.edges(data=True):
            if data.get('congestion_prone', False):
                # Add penalty for congestion-prone edges
                congestion_sum += data['weight']
                edge_count += 1
        
        # Return 0 if no congested edges
        if edge_count == 0:
            return 0
        
        # Return average congestion value, ensuring it's not too large
        return (congestion_sum / edge_count) / 10

    def congestion_heuristic(self, node1, node2):
        """Heuristic considering congestion"""
        x1, y1 = self.graph.positions[node1]
        x2, y2 = self.graph.positions[node2]
        
        # Calculate base distance
        distance = math.hypot(x1 - x2, y1 - y2)
        min_edge_weight = min(data['weight'] for _, _, data in self.graph.G.edges(data=True))
        
        # Scale base distance
        scaled_distance = (distance * min_edge_weight) / 25
        
        # Calculate congestion penalty
        congestion_penalty = self.calculate_congestion_penalty(node1, node2)
        
        # Return scaled distance plus small congestion penalty
        return scaled_distance + congestion_penalty

    def zero_heuristic(self, node1, node2):
        """Zero heuristic (Dijkstra's algorithm)"""
        return 0

    def heuristic(self, node1, node2):
        """Main heuristic function that delegates to the selected heuristic"""
        heuristic_functions = {
            'manhattan_distance': self.manhattan_distance,
            'euclidean_distance': self.euclidean_distance,
            'modified_euclidean_distance': self.modified_euclidean_distance,
            'congestion': self.congestion_heuristic,
            'zero_heuristic': self.zero_heuristic
        }
        return heuristic_functions[self.heuristic_type](node1, node2)