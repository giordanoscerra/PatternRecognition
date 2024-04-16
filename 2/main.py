import numpy as np

# Define the Node of a Bayesian Network
class Node:
    def __init__(self, name, parents=None, probabilities=None, order=None, values= None):
        # name: name of the node
        self.name = name
        # parents: list of the parents of the node
        self.parents = parents if parents else None
        # probabilities: conditional probability table of the node
        self.probabilities = probabilities
        # order: level of the node in the network. The root node has order 0, leaf nodes have the highest order.
        self.order = order
        # values: possible values of the node
        self.values = values

# Define the Bayesian Network
class BayesianNetwork:
    def __init__(self):
        # nodes: list of nodes in the network
        self.nodes = []

    # Add a node to the network
    def add_node(self, node):
        self.nodes.append(node)
        # Update the order of the nodes. Topology and causal ordering are preserved.
        self.nodes.sort(key=lambda x: x.order)

    # Get a node by name
    def get_node(self, name):
        return self.nodes.get(name)
    
    # Print the nodes of the network
    def print_nodes(self):
        for node in self.nodes:
            print('name: ', node.name)
            if(not node.parents):
                print('root node')
                print('CPT: ',(node.probabilities), '\n')
            else:
                print('parents: ', node.parents)
                print('CPT:')
                for key in node.probabilities:
                    print((key))
                print('level: ',node.order, '\n')
            
    
    # Generate a sample from the network
    def ancestral_sample(self):
        # Initialize an empty sample
        sample = {}
        # Initialize the probability of the sample
        probability = 1
        # Iterate over the nodes in the network
        for node in self.nodes:
            # If the node is a root node
            if not node.parents:
                # Sample from the node: choose a value from the possible values of the node according to the CPT of the node
                sample[node.name] = np.random.choice(node.values, p=list(node.probabilities.values()))
            # If the node is not a root node
            else:
                # Get the values of the parents of the node. The values are already in the sample because of the causal ordering of the nodes.
                parent_values = [sample[parent] for parent in node.parents]
                # Create a list of cases to consider in the CPT of the node
                cases = []
                # Iterate over the possible values of the node
                for val in node.values:
                    # Add the case to the list of cases, which is a tuple of the values of the parents and the value of the node
                    cases.append(tuple(parent_values + [val]))
                # Flag to check if the correct case is found
                found = False
                # Iterate over the cases in the CPT of the node
                for case in cases:
                    # If the correct case is found
                    for cpt_entry in node.probabilities:
                      # Correct case
                      if case in list(cpt_entry.keys()):
                        # Sample from the node: choose a value from the possible values of the node according to the CPT of the node
                        sample[node.name] = np.random.choice(node.values, p=list(cpt_entry.values()))
                        # Retrieve the probability of the sample from the CPT of the node
                        output = tuple(parent_values + [sample[node.name]])
                        output_value = cpt_entry[output]
                        # Update the probability of the sample
                        probability *= output_value
                        # Set the flag to True
                        found = True
                    # If the correct case is found, break the loop
                    if found:
                        break

        return sample, probability
    
    # Generate n samples from the network and calculate the mean and standard deviation of the joint probability of the samples
    def run_ancestral_sampling(self, n_samples=10):
        # Generate 10 samples from the network
        stats = []
        for i in range(n_samples):
            # Generate a sample from the network
            sample, probability = self.ancestral_sample()
            # Print the sample
            print('Sample number',i+1,': ')
            for key in sample:
                print(key,': ',sample[key])
            # Print the probability of the sample
            print('Sample probability: ',probability) 
            print("\n")
            stats.append(probability)

        # Calculate the mean of the joint probabilities of the samples
        stats = np.array(stats)
        print('Mean of joint probabilities: ', stats.mean())

###################################################
#########    DEFINITION OF ENDRIGONET     #########                    
###################################################

# cherry_tree is the root node. It has two possible values: Young and Old. Each value has the same probability.
cherry_tree = Node(name="cherry_tree", 
                   probabilities={
                       'Young': 0.5,
                       'Old': 0.5
                   }, order = 0,
                   values = ['Young', 'Old']
                )
# branch is a child of cherry_tree. It has two possible values: Strong and Weak.
# The probability of the branch being Strong is 0.7 if the cherry_tree is Young. This seems reasonable because young trees are usually stronger.
# The probability of the branch being Strong is 0.2 if the cherry_tree is Old. This seems reasonable because old trees are usually weaker.
branch = Node(name="branch",
              parents=["cherry_tree"],
              probabilities=[{
                  ('Young', 'Strong'): 0.7,
                  ('Young', 'Weak'): 0.3
              },
                  {
                      ('Old', 'Strong'): 0.2,
                      ('Old', 'Weak'): 0.8
                  }], order=1,
              values=['Strong', 'Weak']
            )
# flower is a child of branch. It has two possible values: Beautiful and Ugly.
# The probability of the flower being Beautiful is 0.9 if the branch is Strong. This seems reasonable because strong branches are usually healthier.
# The probability of the flower being Beautiful is 0.3 if the branch is Weak. This seems reasonable because weak branches are usually less healthy, but we can still have some beautiful flowers.
flower = Node(name="flower",
              parents=["branch"],
              probabilities=[{
                  ('Strong', 'Beautiful'): 0.9,
                  ('Strong', 'Ugly'): 0.1
              }, {
                  ('Weak', 'Beautiful'): 0.3,
                  ('Weak', 'Ugly'): 0.7
              }], order=2,
              values=['Beautiful', 'Ugly'])
# fruit is a child of flower. It has two possible values: Ripe and Unripe.
# The probability of the fruit being Ripe is 0.1 if the flower is Beautiful. This seems reasonable because beautiful flowers are usually healthier. Endrigo would agree with this, as he agrees that bad things can't grow from good things.
# The probability of the fruit being Ripe is 0.4 if the flower is Ugly. This seems reasonable because ugly flowers are usually less healthy. But we are still on the good side, as we're optimistic that good things can grow from bad things.
fruit = Node(name="fruit",
             parents=["flower"],
             probabilities=[{
                 ('Beautiful', 'Ripe'): 0.1,
                 ('Beautiful', 'Unripe'): 0.9
             }, {
                 ('Ugly', 'Ripe'): 0.4,
                 ('Ugly', 'Unripe'): 0.6
             }],order=3,
             values=['Ripe', 'Unripe'])
# seed is a child of fruit. It has two possible values: Fertile and Infertile.
# The probability of the seed being Infertile is 0.7 if the fruit is Ripe. This seems reasonable because ripe fruits usually have infertile seeds.
# The probability of the seed being Fertile is 0.8 if the fruit is Unripe. This seems reasonable because unripe fruits usually have fertile seeds.
seed = Node(name="seed",
            parents=["fruit"],
            probabilities=[{
                ('Ripe', 'Fertile'): 0.3,
                ('Ripe', 'Infertile'): 0.7
            }, {
                ('Unripe', 'Fertile'): 0.8,
                ('Unripe', 'Infertile'): 0.2
            }],order=4,
            values=['Fertile', 'Infertile'])
# ground is a child of flower. It has two possible values: Fertile and Infertile.
# The probability of the ground being Fertile is 0.8 if the flower is Beautiful. This seems reasonable because beautiful flowers usually grow in fertile ground. And also Endrigo seems quite sure about this one.
# The probability of the ground being Fertile is 0.4 if the flower is Ugly. This seems reasonable because ugly flowers usually grow in infertile ground. But we are still on the good side, as we're optimistic that good things can grow from "bad" things.
ground = Node(name="ground",
                     parents=["flower"],
                     probabilities=[{
                         ('Beautiful', 'Fertile'): 0.8,
                         ('Beautiful', 'Infertile'): 0.2
                     }, {
                         ('Ugly', 'Fertile'): 0.6,
                         ('Ugly', 'Infertile'): 0.4
                     }],order=3,
                     values=['Fertile', 'Infertile'])
# mount is a child of ground. It has three possible values: Wooded, Arid, and Artificial.
# Here there's a twist. If we have fertile ground, we have a higher probability of having a wooded mount. 
# But if we have infertile ground, we have a higher probability of having an artificial mount. We don't want our table industry to fail!
# Arid mounts are not very common, but they can still happen. We have a small probability of having an arid mount, regardless of the fertility of the ground. This is for when we've given up on the table industry.
mount = Node(name="mount",
             parents=["ground"],
             probabilities=[{
                 ('Fertile', 'Wooded'): 0.8,
                 ('Fertile', 'Arid'): 0.05,
                 ('Fertile', 'Artificial'): 0.15
             }, {
                 ('Infertile', 'Wooded'): 0.15,
                 ('Infertile', 'Arid'): 0.05,
                 ('Infertile', 'Artificial'): 0.8
             }],order=4,
             values=['Wooded', 'Arid', 'Artificial'])
# forest is a child of mount. It has two possible values: Thick and Sparse.
# With wooded mounts we have a higher probability of having a thick forest. 
# With artificial mounts we have a slightly higher probability of having a sparse forest. 
# With arid mounts, we almost surely have a sparse forest. Remember here we've sadly given up on the table industry.
forest = Node(name="forest",
              parents=["mount"],
              probabilities=[{
                  ('Wooded', 'Thick'): 0.8,
                  ('Wooded', 'Sparse'): 0.2
              }, {
                  ('Arid', 'Thick'): 0.1,
                  ('Arid', 'Sparse'): 0.9
              }, {
                  ('Artificial', 'Thick'): 0.7,
                  ('Artificial', 'Sparse'): 0.3
                  }],order=5,
              values=['Thick', 'Sparse'])
# pine_tree is a child of seed and forest. It has two possible values: Healthy and Sick.
# We probably have a healthy pine tree if we have fertile seeds.
# Sparseness of the forest has a slightly lower impact on the health of the pine tree. 
pine_tree = Node(name="pine_tree",
                 parents=["seed", "forest"],
                 probabilities=[
                     {('Fertile', 'Thick', 'Healthy'): 0.8,
                      ('Fertile', 'Thick', 'Sick'): 0.2},
                     
                     {('Fertile', 'Sparse', 'Healthy'): 0.7,
                      ('Fertile', 'Sparse', 'Sick'): 0.3},
                     
                     {('Infertile', 'Thick', 'Healthy'): 0.5,
                      ('Infertile', 'Thick', 'Sick'): 0.5},
                     
                     {('Infertile', 'Sparse', 'Healthy'): 0.2,
                      ('Infertile', 'Sparse', 'Sick'): 0.8}
                 ],order=6,
                 values=['Healthy', 'Sick'])
# wood is a child of pine_tree. It has two possible values: Strong and Weak.
# We probably have strong wood if we have a healthy pine tree.
wood = Node(name="wood",
            parents=["pine_tree"],
            probabilities=[{
                ('Healthy', 'Strong'): 0.8,
                ('Healthy', 'Weak'): 0.2
            }, {
                ('Sick', 'Strong'): 0.2,
                ('Sick', 'Weak'): 0.8
            }],order=7,
            values=['Strong', 'Weak'])
# table is a child of wood. It has three possible values: Long Lasting, Fragile, and Broken.
# We probably have a long-lasting table if we have strong wood. There are still chances of having a fragile or broken table, but they are very low.
# We probably have a fragile table if we have weak wood. There are still chances of having a long-lasting table, but they are very low. More likely, we have a broken table.
table = Node(name="table",
             parents=["wood"],
             probabilities=[{
                 ('Strong', 'Long Lasting'): 0.7,
                 ('Strong', 'Fragile'): 0.18,
                 ('Strong', 'Broken'): 0.12
             }, {
                 ('Weak', 'Long Lasting'): 0.1,
                 ('Weak', 'Fragile'): 0.6,
                 ('Weak', 'Broken'): 0.3
             }],order=8,
             values=['Long Lasting', 'Fragile', 'Broken'])

###################################################
#########       NETWORK COMPILATION       #########                    
###################################################

# Initialize the network
EndrigoNet = BayesianNetwork()

# Add nodes to the network in whatever order (topology and causal ordering are preserved)
EndrigoNet.add_node(cherry_tree)
EndrigoNet.add_node(branch)
EndrigoNet.add_node(flower)
EndrigoNet.add_node(fruit)
EndrigoNet.add_node(seed)
EndrigoNet.add_node(ground)
EndrigoNet.add_node(mount)
EndrigoNet.add_node(forest)
EndrigoNet.add_node(pine_tree)
EndrigoNet.add_node(wood)
EndrigoNet.add_node(table)

# Uncomment to print the nodes of the network
# network.print_nodes()

# Generate 100 samples from the network and calculate the mean and standard deviation of the joint probability of the samples
EndrigoNet.run_ancestral_sampling(n_samples=100)

###################################################
#########  DEFINITION OF BEATLESCRASHNET  #########                    
###################################################


producer_death = Node(name="producer_death",
                        probabilities={
                            'Yes': 0.5,
                            'No': 0.5
                        }, order=0,
                        values=['Yes', 'No']
                        )
meeting_of_yoko = Node(name="meeting_of_yoko",
                       probabilities={
                            'Yes': 0.5,
                            'No': 0.5
                          }, order=0,
                            values=['Yes', 'No']
                        )

internal_pressure_economic_problems = Node(name="internal_pressure_economic_problems",
                                           parents=["producer_death"],
                                             probabilities=[{
                                                  ('Yes', 'Yes'): 0.6,
                                                  ('Yes', 'No'): 0.4
                                             }, {
                                                  ('No', 'Yes'): 0.1,
                                                  ('No', 'No'): 0.9
                                             }], order=1,
                                                values=['Yes', 'No']
                                            )

lennon_heroin_use = Node(name="lennon_heroin_use",
                            parents=["meeting_of_yoko"],
                            probabilities=[{
                                ('Yes', 'Yes'): 0.6,
                                ('Yes', 'No'): 0.4
                            }, {
                                ('No', 'Yes'): 0.1,
                                ('No', 'No'): 0.9
                            }], order=1,
                            values=['Yes', 'No']
                            )

beatles_breakup = Node(name="beatles_breakup",
                        parents=["internal_pressure_economic_problems", "lennon_heroin_use"],
                        probabilities=[
                            {('Yes', 'Yes', 'Yes'): 0.9,
                            ('Yes', 'Yes', 'No'): 0.1},
                            
                            {('Yes', 'No', 'Yes'): 0.7,
                            ('Yes', 'No', 'No'): 0.3},
                            
                            {('No', 'Yes', 'Yes'): 0.4,
                            ('No', 'Yes', 'No'): 0.6},
                            
                            {('No', 'No', 'Yes'): 0.1,
                            ('No', 'No', 'No'): 0.9}
                        ], order=2,
                        values=['Yes', 'No']
                        )

BeatlesCrashNet = BayesianNetwork()

BeatlesCrashNet.add_node(producer_death)
BeatlesCrashNet.add_node(meeting_of_yoko)
BeatlesCrashNet.add_node(internal_pressure_economic_problems)
BeatlesCrashNet.add_node(lennon_heroin_use)
BeatlesCrashNet.add_node(beatles_breakup)

BeatlesCrashNet.run_ancestral_sampling(n_samples=3)


# https://drive.google.com/uc?export=view&id=1KYKPoh65BIblZlZ7u-rgaRoMA6yV7Hx3 # BeatlesCrashNet
# https://drive.google.com/uc?export=view&id=12n3DwmyoXwdLATAON6pgLfNNaC-_7DfX # EndrigoNet