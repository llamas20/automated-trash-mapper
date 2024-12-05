# automated-trash-mapper
Final Project for Foundations of AI

python version used: 3.11.9

Required packages are included in the requirements.txt

## To run our code:
_python3 simulation.py_

You can modify these parameters in the simulation.py:

**seed:** the random seed

**num_agents:** the number of agents

**num_nodes:** the number of nodes

**hub_node:** specify which node number to use as a hub

**max_load:** maximum load that an agent can collect

**heuristic_type:** the type of heuristic used in the A* algortithm


## To test our code:
### Test Manhattan distance heuristic
_python3 test_heuristics.py_ manhattan_distance

### Test Euclidean distance heuristic
_python3 test_heuristics.py_ euclidean_distance

### Test modified Euclidean distance heuristic
_python3 test_heuristics.py_ modified_euclidean_distance

### Test congestion heuristic
_python3 test_heuristics.py_ congestion

### Test zero heuristic
_python3 test_heuristics.py_ zero_heuristic

