from map import CustomGraph
from agent import Agent

class AgentTerritoryOrchestrator:
    def __init__(self, graph, num_agents, hub_node=1):
        self.graph = graph # allow graph to be accessed by orchestrator 
        self.agents = [] # this will hold all the agents
        self.hub_node = hub_node # this node is hub for agents. # Maybe we should have this be an attribute of the map, not the agent...

        for i in range(num_agents):
            new_agent = Agent(graph, hub_node, hub_node, name=i)
            self.agents.append(new_agent)

        # TODO 
        # create initial distribution of territory
        # we could potentially just call self.redistribute(), depending on implementation
        
        return
    
    def step(self):
        # determine if redistribution is needed
        if self.check_redistribution_conditions():
            self.redistribute()

        # step all agents
        for agent in self.agents:
            agent.step()
        
        return

    def check_redistribution_conditions(self):
        # TODO create  algorithm here
        return False

    def redistribute(self):
        # TODO create redistribution algorithm here
        return
