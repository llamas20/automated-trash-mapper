import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import random
from scipy.spatial import Delaunay

#Built by ChatGPT
class CustomGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.positions = {}  # Dictionary to store node positions
        self.congested_edges = set()

    def add_node(self, node, x, y):
        """Add a node with x, y coordinates"""
        self.G.add_node(node, pos=(x, y))
        self.positions[node] = (x, y)

    def add_edge(self, node1, node2, weight, targets, congestion_prone=False, label=1):
        """Add an edge with a weight and targets, and mark if it is prone to congestion"""
        if not self.edge_overlap(node1, node2):
            self.G.add_edge(node1, node2, weight=weight, targets=targets, congestion_prone=congestion_prone, base_weight=weight, label=label)
            return True
        else:
            print(f"Edge between {node1} and {node2} overlaps with an existing edge. Skipping.")
            return False

    def edge_overlap(self, node1, node2):
        """Check if the edge between node1 and node2 overlaps with any existing edges, excluding edges that share a node."""
        for u, v in self.G.edges():
            # Skip if edges share a node
            if u in (node1, node2) or v in (node1, node2):
                continue
            if self.do_lines_intersect(self.positions[u], self.positions[v], self.positions[node1], self.positions[node2]):
                return True
        return False

    def do_lines_intersect(self, p1, p2, q1, q2):
        """Check if two line segments (p1, p2) and (q1, q2) intersect"""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    def update_edge_weight(self, node1, node2, new_weight):
        """Update the weight of an edge"""
        if self.G.has_edge(node1, node2):
            self.G[node1][node2]['weight'] = new_weight
        else:
            print(f"No edge exists between {node1} and {node2}")

    def update_edge_targets(self, node1, node2, new_targets):
        """Update the targets value of an edge"""
        if self.G.has_edge(node1, node2):
            self.G[node1][node2]['targets'] = new_targets
        else:
            print(f"No edge exists between {node1} and {node2}")

    def draw_graph_with_agent(self, agents, step_time=0.1):
        """Draw the graph with node positions, edge weights, targets, and agents"""
        plt.clf()  # Clear the current figure

        # Determine edge colors based on congestion
        edge_colors = [
            'red' if (u, v) in self.congested_edges or (v, u) in self.congested_edges else 'gray'
            for u, v in self.G.edges()
        ]

        # Determine node colors and sizes
        node_colors = ['lightblue'] * self.G.number_of_nodes()
        node_sizes = [300] * self.G.number_of_nodes()

        # If there are agents, highlight their positions
        for agent in agents:
            if agent.current_node in self.positions:
                node_index = list(self.G.nodes()).index(agent.current_node)
                node_colors[node_index] = 'orange'
                node_sizes[node_index] = 500

        # Draw nodes
        nx.draw_networkx_nodes(self.G, self.positions, node_color=node_colors, node_size=node_sizes)

        # Draw edges with color coordinated by congestion
        nx.draw_networkx_edges(self.G, self.positions, edge_color=edge_colors, width=2)

        # Draw labels
        nx.draw_networkx_labels(self.G, self.positions, font_size=10, font_color='black')

        # Draw edge labels with weights and targets
        edge_labels = {(u, v): f"W:{self.G[u][v]['weight']} T:{self.G[u][v]['targets']} L:{self.G[u][v]['label']}" for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, self.positions, edge_labels=edge_labels, font_size=8)

        # Draw targets on edges (red dots with size proportional to targets)
        for (u, v) in self.G.edges():
            targets = self.G[u][v]['targets']
            if targets > 0:
                midpoint = np.mean([self.positions[u], self.positions[v]], axis=0)
                plt.scatter(midpoint[0], midpoint[1], s=targets * 50, c='red', alpha=0.6, marker='o')

        # Draw agents on the map as green dots
        for agent in agents:
            if agent.current_node in self.positions:
                agent_pos = self.positions[agent.current_node]
                plt.scatter(agent_pos[0], agent_pos[1], s=300, c='green', edgecolors='black', label=f'Agent @ {agent.current_node}')

        # Handle legend for agents
        if agents:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.axis('off')
        plt.title('Agent Simulation', fontsize=15)
        plt.pause(step_time)  # Pause to update the plot

    def build_a_map(self):
        """Generates a large map using Delaunay triangulation with 50 nodes."""
        num_nodes = 50
        # Generate random positions for the nodes
        positions = {}
        for node_id in range(1, num_nodes + 1):
            x = random.randint(0, 1000)
            y = random.randint(0, 1000)
            positions[node_id] = (x, y)
            self.add_node(node_id, x, y)

        # Create a list of points for Delaunay triangulation
        points = np.array(list(positions.values()))
        # Perform Delaunay triangulation
        tri = Delaunay(points)
        # Mapping from point indices to node IDs
        indices = list(positions.keys())
        # Extract edges from the triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                idx1 = simplex[i]
                idx2 = simplex[(i + 1) % 3]
                node1 = indices[idx1]
                node2 = indices[idx2]
                edge = tuple(sorted((node1, node2)))
                edges.add(edge)

        # Randomly select some edges as congestion-prone
        num_congestion_prone_edges = int(len(edges) * 0.2)  # For example, 20% of edges
        congestion_prone_edges = random.sample(list(edges), num_congestion_prone_edges)

        # Calculate the maximum possible distance in the grid
        max_distance = np.hypot(1000, 1000)  # Approximately 1414.21

        # Add the edges to the graph
        for edge in edges:
            node1, node2 = edge
            # Calculate distance
            x1, y1 = self.positions[node1]
            x2, y2 = self.positions[node2]
            distance = np.hypot(x2 - x1, y2 - y1)

            # Normalize the weight to be between 1 and 20
            normalized_weight = int(round((distance / max_distance) * 20))
            # Ensure the weight is at least 1
            normalized_weight = max(1, normalized_weight)

            # Initial targets
            targets = random.randint(1, 10)

            # Check if this edge is congestion-prone
            congestion_prone = edge in congestion_prone_edges

            # TODO:  Map subdivisions
            label = np.random.randint(1, 4)

            edge_added = self.add_edge(node1, node2, weight=normalized_weight, targets=targets,
                                       congestion_prone=congestion_prone, label=label)
            if edge_added:
                # Edge was added, safe to set base_weight
                self.G[node1][node2]['base_weight'] = normalized_weight

    def update_map(self):
        """Update each edge with new congestion levels and targets."""
        # Update each edge with new congestion levels
        new_weights = {}
        congested_edges = set()

        for node1, node2 in self.G.edges():
            edge_data = self.G[node1][node2]
            base_weight = edge_data['base_weight']

            # Simulate congestion factor
            if edge_data.get('congestion_prone', False):  # If congestion-prone
                congestion_factor = random.uniform(1.0, 4.0)  # Higher congestion factor
            else:
                congestion_factor = random.uniform(0.4, 1.6)

            # Check if neighboring edges are congested
            neighboring_edges = self.get_neighboring_edges(node1, node2)
            for neighbor in neighboring_edges:
                neighbor_weight = self.G[neighbor[0]][neighbor[1]]['weight']
                if neighbor_weight > 10:
                    # Neighboring edge is congested, increase congestion factor
                    congestion_factor *= 1.2

            # Update weight
            new_weight = int(round(base_weight * congestion_factor))
            # Ensure the new weight is between 1 and 20
            new_weight = max(1, min(new_weight, 20))
            new_weights[(node1, node2)] = new_weight

            # If weight exceeds threshold, mark as congested
            if new_weight > 10:
                congested_edges.add((node1, node2))
            else:
                # If edge is no longer congested, remove it from congested_edges
                self.congested_edges.discard((node1, node2))

        # Update weights in the graph
        for (node1, node2), new_weight in new_weights.items():
            self.update_edge_weight(node1, node2, new_weight)

        # Update congested edges
        self.congested_edges.update(congested_edges)

        # Update targets
        for node1, node2 in self.G.edges():
            current_targets = self.G[node1][node2]['targets']
            increment = random.randint(0, 2)
            max_targets = 20
            new_targets = min(current_targets + increment, max_targets)
            self.update_edge_targets(node1, node2, new_targets)

    def get_neighboring_edges(self, node1, node2):
        """Get edges neighboring the edge (node1, node2)"""
        neighboring_edges = set()
        nodes = {node1, node2}
        for node in nodes:
            for neighbor in self.G.neighbors(node):
                if neighbor != node1 and neighbor != node2:
                    edge = tuple(sorted((node, neighbor)))
                    neighboring_edges.add(edge)
        return neighboring_edges
