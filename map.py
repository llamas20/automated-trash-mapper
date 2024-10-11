import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import itertools
from random import randint

# Built by ChatGPT
class CustomGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.positions = {}  # Dictionary to store node positions

    def add_node(self, node, x, y):
        """Add a node with x, y coordinates"""
        self.G.add_node(node, pos=(x, y))
        self.positions[node] = (x, y)

    def add_edge(self, node1, node2, weight, targets):
        """Add an edge with a weight and targets, ensuring no overlap"""
        if not self.edge_overlap(node1, node2):
            self.G.add_edge(node1, node2, weight=weight, targets=targets)
        else:
            print(f"Edge between {node1} and {node2} overlaps with an existing edge. Skipping.")

    def edge_overlap(self, node1, node2):
        """Check if the edge between node1 and node2 overlaps with any existing edges"""
        for u, v in self.G.edges():
            if self.do_lines_intersect(self.positions[u], self.positions[v], self.positions[node1], self.positions[node2]):
                return True
        return False

    def do_lines_intersect(self, p1, p2, q1, q2):
        """Check if two line segments (p1, p2) and (q1, q2) intersect"""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

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

    def draw_graph(self):
        """Draw the graph with node positions and edge weights"""
        plt.figure()
        nx.draw(self.G, self.positions, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        
        # Draw edge labels with weights and targets
        edge_labels = {(u, v): f"W:{self.G[u][v]['weight']} T:{self.G[u][v]['targets']}" for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, self.positions, edge_labels=edge_labels)
        
        plt.show()
    
    # TODO
    def build_a_map(self):
        # Adding nodes with x, y coordinates
        self.add_node(1, 0, 0)
        self.add_node(2, 2, 3)
        self.add_node(3, 4, 0)
        self.add_node(4, 1, -2)

        # Adding edges with weight and targets
        self.add_edge(1, 2, weight=5, targets=3)
        self.add_edge(2, 3, weight=3, targets=2)
        self.add_edge(1, 4, weight=7, targets=4)
        self.add_edge(3, 4, weight=6, targets=1)  # Should avoid overlap

        return
    

    # TODO
    def update_map(self):
        # Updating an edge
        self.update_edge_weight(1, 2, new_weight=randint(1,10))
        self.update_edge_targets(1, 2, new_targets=randint(1,10))
        self.update_edge_weight(2, 3, new_weight=randint(1,10))
        self.update_edge_targets(2, 3, new_targets=randint(1,10))