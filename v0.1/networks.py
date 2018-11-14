"""
networks.py
A module that contains all the networks architecture classes.
Define layers or combination of layers here.
"""
import numpy as np
import neuron_models as nm
import networkx as nx
import matplotlib.pyplot as plt

""" The most basic class"""
class LayeredDiGraph(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)
        self.layers = [self]
        self.labels = []

"""
Populate a single layer of num_neruon NeuronClass.
"""
def get_single_layer(NeuronClass, num_neuron):
    sl = LayeredDiGraph()
    # populate instances of NeuronClass
    sl.add_nodes_from([NeuronClass() for n in range(num_neuron)])
    sl.labels = ["Single Layer"]
    return sl

"""
fully_connect(net1, net2, SynapseClass):

Return a network with the last layer of net1 fully connect onto the top layer
# of net2 with SynapseClass.

Here "top" and "last" refer to the order in net*.layers.
"""
def fully_connect(net1, net2, SynapseClass):
    net3 = nx.compose(net1, net2)
    pre_neurons, pos_neurons = net1.layers[-1].nodes(), net2.layers[0].nodes()
    for pos_neuron in pos_neurons:
        for pre_neuron in pre_neurons:
            # order matters when adding edges
            net3.add_edge(pre_neuron, pos_neuron, synapse=SynapseClass())
    net3.layers = net1.layers + net2.layers
    net3.labels = net1.labels + net2.labels
    return net3

def get_multilayer_fc(NeuronClass, SynapseClass, neuron_nums):
    num_layer = len(neuron_nums)
    net = get_single_layer(NeuronClass, neuron_nums[0])
    for i in range(1, num_layer):
        sl = get_single_layer(NeuronClass, neuron_nums[i])
        net = fully_connect(net, sl, SynapseClass)
    return net


"""
Draw layered graph in a structured manner.
"""
def draw_layered_digraph(net):
    num_layer = len(net.layers)
    xs = np.linspace(0., 1., num=num_layer)
    pos = {}
    for (l, layer) in enumerate(net.layers):
        if len(layer.nodes) == 1:
            ys = [0.5]
        else:
            ys = np.linspace(0., 1., num=len(layer.nodes))
        for (n, neuron) in enumerate(layer.nodes):
            pos[neuron] = [xs[l], ys[n]]
    plt.figure()
    nx.draw_networkx(net, pos=pos, with_labels=False)
