import unittest
from agent import Agent
from map import CustomGraph
import networkx as nx
import numpy as np
import sys

# Built by ChatGPT
class TestHeuristic(unittest.TestCase):
    """Test class for heuristic functions"""
    
    @classmethod
    def setUpClass(cls):
        # Get heuristic type from command line arguments
        if len(sys.argv) > 1:
            cls.heuristic_type = sys.argv[1]  # Get heuristic type
            # Remove parameter to avoid unittest parsing it
            sys.argv.pop()
        else:
            cls.heuristic_type = 'manhattan_distance'  # Default heuristic
        print(f"\nTesting {cls.heuristic_type} heuristic...")

    def setUp(self):
        self.graph = CustomGraph()
        
        # Add test nodes with specific coordinates
        self.graph.add_node(1, 0, 0)
        self.graph.add_node(2, 3, 4)
        self.graph.add_node(3, 6, 8)
        self.graph.add_node(4, 2, 10)
        
        # Add test edges with specific weights
        self.graph.add_edge(1, 2, 5, 0)  # weight=5
        self.graph.add_edge(2, 3, 6, 0)  # weight=6
        self.graph.add_edge(3, 4, 4, 0)  # weight=4
        self.graph.add_edge(1, 4, 12, 0) # weight=12
        
        # Create agent with the heuristic type from class variable
        self.agent = Agent(self.graph, 1, 1, 1, 1000, self.__class__.heuristic_type)

    def test_admissibility(self):
        """Test the admissibility of the heuristic function"""
        print(f"\nTesting {self.__class__.heuristic_type} admissibility...")
        
        for start_node in self.graph.G.nodes():
            for goal_node in self.graph.G.nodes():
                if start_node != goal_node:
                    h_value = self.agent.heuristic(start_node, goal_node)
                    
                    try:
                        actual_path = nx.shortest_path(self.graph.G, start_node, goal_node, weight='weight')
                        actual_cost = sum(self.graph.G[actual_path[i]][actual_path[i+1]]['weight'] 
                                        for i in range(len(actual_path)-1))
                        
                        print(f"From node {start_node} to node {goal_node}:")
                        print(f"Heuristic value: {h_value:.2f}")
                        print(f"Actual cost: {actual_cost}")
                        
                        self.assertLessEqual(h_value, actual_cost, 
                            f"{self.__class__.heuristic_type}: h({start_node},{goal_node})={h_value:.2f} > actual_cost={actual_cost}")
                        
                    except nx.NetworkXNoPath:
                        print(f"No path found from node {start_node} to node {goal_node}")

    def test_consistency(self):
        """Test the consistency of the heuristic function"""
        print(f"\nTesting {self.__class__.heuristic_type} consistency...")
        
        for goal_node in self.graph.G.nodes():
            for edge in self.graph.G.edges():
                n, n_prime = edge
                h_n = self.agent.heuristic(n, goal_node)
                h_n_prime = self.agent.heuristic(n_prime, goal_node)
                edge_cost = self.graph.G[n][n_prime]['weight']
                
                print(f"Goal node: {goal_node}, Edge: {n}->{n_prime}")
                print(f"h({n}) = {h_n:.2f}")
                print(f"h({n_prime}) = {h_n_prime:.2f}")
                print(f"Edge cost = {edge_cost}")
                print(f"h({n}) ≤ {edge_cost} + h({n_prime}) = {edge_cost + h_n_prime:.2f}")
                
                self.assertLessEqual(
                    h_n, 
                    edge_cost + h_n_prime,
                    f"{self.__class__.heuristic_type}: Consistency violation: h({n})={h_n:.2f} > {edge_cost} + h({n_prime})={h_n_prime:.2f}"
                )

    def test_heuristic_distance_accuracy(self):
        """Test the accuracy of heuristic distance calculation"""
        print(f"\nTesting {self.__class__.heuristic_type} distance calculation...")
        
        # Test points with known distances
        node1, node2 = 1, 2  # (0,0) to (3,4)
        
        # Calculate expected distance based on heuristic type
        min_edge_weight = min(data['weight'] for _, _, data in self.graph.G.edges(data=True))
        
        if self.__class__.heuristic_type == 'manhattan_distance':
            manhattan_distance = abs(3 - 0) + abs(4 - 0)  # = 7
            expected_distance = (manhattan_distance * min_edge_weight) / 10
        elif self.__class__.heuristic_type == 'euclidean_distance':
            euclidean_distance = 5.0  # sqrt(3^2 + 4^2) = 5
            expected_distance = (euclidean_distance * min_edge_weight) / 25
        elif self.__class__.heuristic_type == 'modified_euclidean_distance':
            euclidean_distance = 5.0
            expected_distance = (euclidean_distance * min_edge_weight) / 15
        elif self.__class__.heuristic_type == 'zero_heuristic':
            expected_distance = 0
        else:  # congestion
            # Skip exact comparison for congestion as it depends on dynamic factors
            return
        
        calculated_distance = self.agent.heuristic(node1, node2)
        print(f"Node {node1} to node {node2}:")
        print(f"Calculated distance: {calculated_distance:.2f}")
        print(f"Expected distance: {expected_distance:.2f}")
        
        if self.__class__.heuristic_type != 'congestion':
            self.assertAlmostEqual(
                calculated_distance, 
                expected_distance, 
                places=2,
                msg=f"{self.__class__.heuristic_type}: Heuristic distance calculation is inaccurate"
            )

if __name__ == '__main__':
    # Run tests
    unittest.main(argv=['first-arg-is-ignored']) 