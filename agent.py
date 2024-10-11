import random as random

# built by ChatGPT    
class Agent:
    def __init__(self, graph, start_node, load=0):
        """Initialize agent on the graph at a starting node"""
        self.graph = graph  # The CustomGraph instance 
        self.current_node = start_node  # The current node the agent is at
        if start_node not in graph.G:
            raise ValueError(f"Node {start_node} does not exist in the graph")
        self.load = load

    #TODO
    def step(self):
        """Move the agent to a neighboring node along an edge"""
        neighbors = list(self.graph.G.neighbors(self.current_node))
        if not neighbors:
            print(f"No neighbors to move to from node {self.current_node}")
            return

        # Choose a random neighbor to move to for simplicity
        next_node = random.choice(neighbors)
        print(f"Moving from node {self.current_node} to node {next_node}")
        self.current_node = next_node

    def get_position(self):
        """Return the current node of the agent"""
        return self.current_node