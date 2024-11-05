import networkx as nx
from map import CustomGraph

#Built by ChatGPT
class Agent:
    def __init__(self, graph, start_node, hub_node, agent_id):
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

        if start_node not in graph.G:
            raise ValueError(f"Node {start_node} does not exist in the graph")
        if hub_node not in graph.G:
            raise ValueError(f"Hub node {hub_node} does not exist in the graph")
        self.collected_targets = 0  # Total targets collected

    def heuristic(self, node1, node2):
        """Heuristic function for A* (Euclidean distance)."""
        x1, y1 = self.graph.positions[node1]
        x2, y2 = self.graph.positions[node2]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

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
        """
        Perform one step of the agent's action:
        - Find the nearest edge with targets that matches agent's ID
        - Move along the path to that edge
        - Traverse the edge to collect targets
        """
        if self.all_targets_collected():
            print(f"Agent {self.agent_id}: All assigned targets collected. At the hub.")
            if self.current_node != self.hub_node:
                self.return_to_hub()
            return False  # Simulation ends

        # Find the nearest edge with targets
        nearest_edge, path_to_edge_start = self.find_nearest_target_edge()
        if not nearest_edge:
            print(f"Agent {self.agent_id}: No edges with targets remaining for this agent.")
            if self.current_node != self.hub_node:
                self.return_to_hub()
            return False  # Simulation ends

        edge_start, edge_end = nearest_edge

        # Move to the start of the edge
        if self.current_node != edge_start:
            path = path_to_edge_start
            print(f"Agent {self.agent_id} moving from {self.current_node} to {edge_start} via path: {path}")
            for node in path[1:]:
                self.current_node = node
                print(f"Agent {self.agent_id} moved to node {self.current_node}")

        # Traverse the edge to collect targets
        if self.current_node == edge_start:
            self.traverse_edge(edge_start, edge_end)

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

    def return_to_hub(self):
        """Find and traverse the path back to the hub node."""
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