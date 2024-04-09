import numpy as np

class Node:
    def __init__(self, name, parents=None):
        self.name = name
        self.parents = parents if parents else []
        self.probabilities = {}

    def add_probability(self, values, probability):
        self.probabilities[tuple(values)] = probability

    def get_probability(self, values):
        return self.probabilities.get(tuple(values), 0.0)

class BayesianNetwork:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.name] = node

    def get_node(self, name):
        return self.nodes.get(name)

    def calculate_probability(self, node_name, node_value, evidence):
        node = self.get_node(node_name)
        if not node:
            return 0.0

        parents = [self.get_node(parent_name) for parent_name in node.parents]
        parent_values = [evidence[parent_name] for parent_name in node.parents]

        probability = node.get_probability([node_value] + parent_values)
        parent_probabilities = [self.calculate_probability(parent.name, parent_value, evidence) for parent, parent_value in zip(parents, parent_values)]

        return probability * np.prod(parent_probabilities)

# Example usage
network = BayesianNetwork()

# Define nodes
flower_tree = Node("flower_tree")
branch = Node("branch", parents=["flower_tree"])
flower = Node("flower", parents=["branch"])
fruit = Node("fruit", parents=["flower"])
seed = Node("seed", parents=["fruit"])
ground = seed = Node("ground", parents=["flower"])
mount = Node("mount", parents=["ground"])
forest = Node("forest", parents=["mount"])
wood_tree = Node("wood_tree", parents=["seed", "forest"])
wood = Node("wood", parents=["wood_tree"])
table = Node("table", parents=["wood"])


# Add nodes to the network
network.add_node(flower_tree)
network.add_node(branch)
network.add_node(flower)
network.add_node(fruit)
network.add_node(seed)
network.add_node(ground)
network.add_node(mount)
network.add_node(forest)
network.add_node(wood_tree)
network.add_node(wood)
network.add_node(table)

def flower_tree_probability(flower_tree):
    # flower tree's age
    return np.random.normal(5, 1)

def branch_probability(branch, flower_tree):
    # if the flower tree is old enough, it will have more branches
    if flower_tree >= 5:
        return np.random.normal(4, 0.5)
    # if the flower tree is too young, it will have less branches
    else:
        return np.random.normal(1, 0.5)
    
def flower_probability(flower, branch):
    # if the branch is big, it will have more flowers
    if branch >= 3:
        return np.random.normal(10, 1)
    # if the branch is small, it will have less flowers
    else:
        return np.random.normal(1, 0.5)
    
def fruit_probability(fruit, flower):
    # if there are many flowers, there'll be more fruits
    if flower >= 5:
        return np.random.normal(5, 1)
    # if the flowers are few, there'll be less fruits
    else:
        return np.random.normal(1, 0.5)

def seed_probability(seed, fruit):
    # if there are many fruits, there'll be more seeds
    if fruit >= 5:
        return np.random.normal(10, 1)
    # if the fruits are few, there'll be less seeds
    else:
        return np.random.normal(1, 0.5)

def ground_probability(ground, flower):
    # if there are many flowers, there'll be more ground    
    if flower >= 5:
        return np.random.normal(10, 1)
    # if the flowers are few, there'll be less ground
    else:
        return np.random.normal(1, 0.5)
    
def mount_probability(mount, ground):
    # if there is a lot of ground, there'll be more mounts
    if ground >= 5:
        return np.random.normal(10, 1)
    # if the ground is scarce, there'll be less mounts
    else:
        return np.random.normal(1, 0.5)
    
def forest_probability(forest, mount):
    # if there are many mounts, there'll be more forests
    if mount >= 5:
        return np.random.normal(10, 1)
    # if the mounts are few, there'll be less forests
    else:
        return np.random.normal(1, 0.5)
    
def wood_tree_probability(wood_tree, seed, forest):
    # if there are many seeds and forests, there'll be more wood trees
    if seed >= 5 and forest >= 5:
        return np.random.normal(10, 1)
    # if the seeds or forests are few, there'll be less wood trees
    else:
        return np.random.normal(1, 0.5)
    
def wood_probability(wood, wood_tree):
    # if there are many wood trees, there'll be more wood
    if wood_tree >= 5:
        return np.random.normal(10, 1)
    # if the wood trees are few, there'll be less wood
    else:
        return np.random.normal(1, 0.5)
    
def table_probability(table, wood):
    # if there is a lot of wood, there'll be more tables
    if wood >= 5:
        return np.random.normal(10, 1)
    # if the wood is scarce, there'll be less tables
    else:
        return np.random.normal(1, 0.5)
