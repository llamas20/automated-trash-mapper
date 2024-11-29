import unittest
from agent import Agent
from map import CustomGraph
import networkx as nx
import numpy as np

class TestAgentHeuristic(unittest.TestCase):
    def setUp(self):
        # Create a simple test graph
        self.graph = CustomGraph()
        
        # Add test nodes
        self.graph.add_node(1, 0, 0)
        self.graph.add_node(2, 3, 4)
        self.graph.add_node(3, 6, 8)
        self.graph.add_node(4, 2, 10)
        
        # Add test edges
        self.graph.add_edge(1, 2, 5, 0)  # weight=5
        self.graph.add_edge(2, 3, 6, 0)  # weight=6
        self.graph.add_edge(3, 4, 4, 0)  # weight=4
        self.graph.add_edge(1, 4, 12, 0) # weight=12
        
        # Create test agent
        self.agent = Agent(self.graph, 1, 1, 1, 1000)

    def test_admissibility(self):
        """
        Test the admissibility of the heuristic function
        Heuristic value should never exceed the actual shortest path cost
        """
        print("\nTesting heuristic function admissibility...")
        
        for start_node in self.graph.G.nodes():
            for goal_node in self.graph.G.nodes():
                if start_node != goal_node:
                    # Calculate heuristic value
                    h_value = self.agent.heuristic(start_node, goal_node)
                    
                    # Calculate actual shortest path cost
                    try:
                        actual_path = nx.shortest_path(self.graph.G, 
                                                     start_node, 
                                                     goal_node, 
                                                     weight='weight')
                        actual_cost = sum(self.graph.G[actual_path[i]][actual_path[i+1]]['weight'] 
                                        for i in range(len(actual_path)-1))
                        
                        print(f"From node {start_node} to node {goal_node}:")
                        print(f"Heuristic value: {h_value:.2f}")
                        print(f"Actual cost: {actual_cost}")
                        
                        # Verify heuristic value is not greater than actual cost
                        self.assertLessEqual(h_value, actual_cost, 
                            f"Heuristic value ({h_value}) is greater than actual cost ({actual_cost})")
                        
                    except nx.NetworkXNoPath:
                        print(f"No path found from node {start_node} to node {goal_node}")

    def test_consistency(self):
        """
        Test the consistency of the heuristic function
        For any two adjacent nodes n and n', and goal node:
        h(n) ≤ cost(n,n') + h(n')
        """
        print("\nTesting heuristic function consistency...")
        
        for goal_node in self.graph.G.nodes():
            for edge in self.graph.G.edges():
                n, n_prime = edge
                
                # Calculate h(n) and h(n')
                h_n = self.agent.heuristic(n, goal_node)
                h_n_prime = self.agent.heuristic(n_prime, goal_node)
                
                # Get edge cost
                edge_cost = self.graph.G[n][n_prime]['weight']
                
                print(f"Goal node: {goal_node}, Edge: {n}->{n_prime}")
                print(f"h({n}) = {h_n:.2f}")
                print(f"h({n_prime}) = {h_n_prime:.2f}")
                print(f"Edge cost = {edge_cost}")
                print(f"h({n}) ≤ {edge_cost} + h({n_prime}) = {edge_cost + h_n_prime:.2f}")
                
                # Verify consistency condition
                self.assertLessEqual(
                    h_n, 
                    edge_cost + h_n_prime,
                    f"Consistency violation: h({n})={h_n} > {edge_cost} + h({n_prime})={h_n_prime}"
                )

    def test_heuristic_distance_accuracy(self):
        """
        Test the accuracy of heuristic distance calculation
        """
        print("\nTesting heuristic distance calculation...")
        
        # Test points with known distance
        node1, node2 = 1, 2  # (0,0) to (3,4)
        manhattan_distance = abs(3 - 0) + abs(4 - 0)  # = 7
        min_edge_weight = min(data['weight'] for _, _, data in self.graph.G.edges(data=True))
        expected_distance = (manhattan_distance * min_edge_weight) / 10  # Use same calculation method
        
        calculated_distance = self.agent.heuristic(node1, node2)
        print(f"Node {node1} to node {node2}:")
        print(f"Calculated distance: {calculated_distance:.2f}")
        print(f"Expected distance: {expected_distance:.2f}")
        
        self.assertAlmostEqual(
            calculated_distance, 
            expected_distance, 
            places=2,
            msg="Heuristic distance calculation is inaccurate"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2) 