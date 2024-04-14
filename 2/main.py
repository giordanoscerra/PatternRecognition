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
    return np.random.normal(10, 2)

def branch_probability(branch, flower_tree):
    # if the flower tree is old enough, it will have more branches
    if flower_tree >= 10:
        return np.random.normal(8, 1)
    # if the flower tree is too young, it will have less branches
    else:
        return np.random.normal(2, 1)
    
def flower_probability(flower, branch):
    # if the branch is big, it will have more flowers
    if branch >= 5:
        return np.random.normal(20, 2)
    # if the branch is small, it will have less flowers
    else:
        return np.random.normal(5, 1)
    
def fruit_probability(fruit, flower):
    # if there are many flowers, there'll be more fruits
    if flower >= 10:
        return np.random.normal(15, 2)
    # if the flowers are few, there'll be less fruits
    else:
        return np.random.normal(5, 1)

def seed_probability(seed, fruit):
    # if there are many fruits, there'll be more seeds
    if fruit >= 10:
        return np.random.normal(20, 2)
    # if the fruits are few, there'll be less seeds
    else:
        return np.random.normal(5, 1)

def ground_probability(ground, flower):
    # if there are many flowers, there'll be more ground    
    if flower >= 10:
        return np.random.normal(20, 2)
    # if the flowers are few, there'll be less ground
    else:
        return np.random.normal(5, 1)
    
def mount_probability(mount, ground):
    # if there is a lot of ground, there'll be more mounts
    if ground >= 10:
        return np.random.normal(20, 2)
    # if the ground is scarce, there'll be less mounts
    else:
        return np.random.normal(5, 1)
    
def forest_probability(forest, mount):
    # if there are many mounts, there'll be more forests
    if mount >= 10:
        return np.random.normal(20, 2)
    # if the mounts are few, there'll be less forests
    else:
        return np.random.normal(5, 1)
    
def wood_tree_probability(wood_tree, seed, forest):
    # if there are many seeds and forests, there'll be more wood trees
    if seed >= 10 and forest >= 10:
        return np.random.normal(20, 2)
    # if the seeds or forests are few, there'll be less wood trees
    else:
        return np.random.normal(5, 1)
    
def wood_probability(wood, wood_tree):
    # if there are many wood trees, there'll be more wood
    if wood_tree >= 10:
        return np.random.normal(20, 2)
    # if the wood trees are few, there'll be less wood
    else:
        return np.random.normal(5, 1)
    
def table_probability(table, wood):
    # if there is a lot of wood, there'll be more tables
    if wood >= 10:
        return np.random.normal(20, 2)
    # if the wood is scarce, there'll be less tables
    else:
        return np.random.normal(5, 1)


'''
import numpy as np

def flower_tree_probability(flower_tree):
    # Flower tree's age categories: Young (0) and Old (1)
    age_category = 1 if flower_tree >= 10 else 0
    # Define probabilities for each age category
    probabilities = [0.1, 0.9]  # Probability of being young or old
    # Sample from multinomial distribution based on age category
    return np.random.multinomial(1, probabilities).argmax()

def branch_probability(branch, flower_tree):
    # Define branch categories: Small (0) and Big (1)
    if flower_tree == 0:
        return np.random.normal(2, 1)
    else:
        size_category = 1 if branch >= 5 else 0
        # Define probabilities for each branch size category
        probabilities = [0.2, 0.8]  # Probability of being small or big
        # Sample from multinomial distribution based on branch size category
        return np.random.multinomial(1, probabilities).argmax()

def flower_probability(flower, branch):
    # Define flower categories: Few (0) and Many (1)
    size_category = 1 if branch >= 5 else 0
    # Define probabilities for each flower size category
    probabilities = [0.2, 0.8]  # Probability of few or many flowers
    # Sample from multinomial distribution based on flower size category
    return np.random.multinomial(1, probabilities).argmax()

def fruit_probability(fruit, flower):
    # Define fruit categories: Few (0) and Many (1)
    size_category = 1 if flower >= 10 else 0
    # Define probabilities for each fruit size category
    probabilities = [0.2, 0.8]  # Probability of few or many fruits
    # Sample from multinomial distribution based on fruit size category
    return np.random.multinomial(1, probabilities).argmax()

def seed_probability(seed, fruit):
    # Define seed categories: Few (0) and Many (1)
    size_category = 1 if fruit >= 10 else 0
    # Define probabilities for each seed size category
    probabilities = [0.2, 0.8]  # Probability of few or many seeds
    # Sample from multinomial distribution based on seed size category
    return np.random.multinomial(1, probabilities).argmax()

def ground_probability(ground, flower):
    # Define ground categories: Little (0) and Much (1)
    size_category = 1 if flower >= 10 else 0
    # Define probabilities for each ground size category
    probabilities = [0.2, 0.8]  # Probability of little or much ground
    # Sample from multinomial distribution based on ground size category
    return np.random.multinomial(1, probabilities).argmax()

def mount_probability(mount, ground):
    # Define mount categories: Few (0) and Many (1)
    size_category = 1 if ground >= 10 else 0
    # Define probabilities for each mount size category
    probabilities = [0.2, 0.8]  # Probability of few or many mounts
    # Sample from multinomial distribution based on mount size category
    return np.random.multinomial(1, probabilities).argmax()

def forest_probability(forest, mount):
    # Define forest categories: Few (0) and Many (1)
    size_category = 1 if mount >= 10 else 0
    # Define probabilities for each forest size category
    probabilities = [0.2, 0.8]  # Probability of few or many forests
    # Sample from multinomial distribution based on forest size category
    return np.random.multinomial(1, probabilities).argmax()

def wood_tree_probability(wood_tree, seed, forest):
    # Define wood tree categories: Few (0) and Many (1)
    size_category = 1 if seed >= 10 and forest >= 10 else 0
    # Define probabilities for each wood tree size category
    probabilities = [0.2, 0.8]  # Probability of few or many wood trees
    # Sample from multinomial distribution based on wood tree size category
    return np.random.multinomial(1, probabilities).argmax()

def wood_probability(wood, wood_tree):
    # Define wood categories: Little (0) and Much (1)
    size_category = 1 if wood_tree >= 10 else 0
    # Define probabilities for each wood size category
    probabilities = [0.2, 0.8]  # Probability of little or much wood
    # Sample from multinomial distribution based on wood size category
    return np.random.multinomial(1, probabilities).argmax()

def table_probability(table, wood):
    # Define table categories: Few (0) and Many (1)
    size_category = 1 if wood >= 10 else 0
    # Define probabilities for each table size category
    probabilities = [0.2, 0.8]  # Probability of few or many tables
    # Sample from multinomial distribution based on table size category
    return np.random.multinomial(1, probabilities).argmax()

'''