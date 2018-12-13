"""
networks.py
A module that contains all the networks architecture classes.
Define layers or combination of layers here.
"""

import numpy as np
import matplotlib.pyplot as plt
# might be useful for mac user, uncommnet below if needed
# import matplotlib
# matplotlib.use("TKAgg")
import random

import neuron_models as nm
import networkx as nx



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

"""
fully_connect_between_layer(net, layer_idxs, SynapseClass):

Unlike fully_connect(), it adds synapse(s) to the network from
layer layer_idxs[0] to layer layer_idxs[1], instead of returning a new network.
i.e input net will be modified.
"""

def fully_connect_between_layer(net, layer_idxs, SynapseClass):
    if len(layer_idxs) != 2:
        print("Expected 2 layers as input, got {}".format(len(layer_idxs)))
    i, j = layer_idxs
    pre_neurons, pos_neurons = net.layers[i].nodes(), net.layers[j].nodes()
    for pos_neuron in pos_neurons:
        for pre_neuron in pre_neurons:
            # order matters when adding edges
            net.add_edge(pre_neuron, pos_neuron, synapse=SynapseClass())

"""
stack(net1, net2):

Return a network with net1 and net2 stacked
No connection is made. (The returned graph will be block-diagonal.)

"""
def stack(net1, net2):
    net3 = nx.compose(net1, net2)
    net3.layers = net1.layers + net2.layers
    net3.labels = net1

"""
get_multilayer_fc(NeuronClass, SynapseClass, neuron_nums):

Return a multilayer fully-connected feedforward network.

"""
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

def draw_colored_layered_digraph(net):
    plt.figure()
    #cycle not working
    cycol = cycle('rbgkc')
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
        print('called')
        nx.draw_networkx_nodes(list(layer.nodes),pos=pos,node_color=next(cycol))
    nx.draw_networkx_edges(net,pos=pos)


"""
Antenna Lobe Functions
-------------------------------------------------------------
"""



def connect_layer(layer, connections, prob, g):
    for n in layer:
        for j in layer:
            if n != j and random.random() < prob:
                layer.add_edge(n, j, synapse = connections(g))

def interconnect(layer1, layer2, synapse1, synapse2, prob1, prob2, g1, g2):
    net3 = nx.compose(layer1, layer2)
    neurons1, neurons2 = layer1.layers[-1].nodes(), layer2.layers[0].nodes()
    for neuron1 in neurons1:
        for neuron2 in neurons2:
            if random.random() < prob1:
                net3.add_edge(neuron1, neuron2, synapse = synapse1(g1))
            if random.random() < prob2:
                net3.add_edge(neuron2, neuron1, synapse = synapse2(g2))
    net3.layers = layer1.layers + layer2.layers
    net3.labels = layer1.labels + layer2.labels
    return net3

#specifically for the 6PN, 2LN network in Fig1 Bazhenov 2001
def manual_connect(LNs, PNs, LNSynapse, PNSynapse, gLN = 400.0, gLNPN = 800.0, gPN = 350.0, gPNLN = 300.0 ):
    #test values
    gLN = 110.0 #400
    gLNPN = 400.0 #800


    #connect LNs together
    connect_layer(LNs, LNSynapse, 1.0, gLN)

    p = list(PNs.nodes())
    #connect PNs together
    PNs.add_edge(p[0], p[1], synapse = PNSynapse(gPN))
    PNs.add_edge(p[0], p[3], synapse = PNSynapse(gPN))
    PNs.add_edge(p[1], p[5], synapse = PNSynapse(gPN))
    PNs.add_edge(p[2], p[0], synapse = PNSynapse(gPN))
    PNs.add_edge(p[2], p[1], synapse = PNSynapse(gPN))
    PNs.add_edge(p[2], p[4], synapse = PNSynapse(gPN))
    PNs.add_edge(p[4], p[3], synapse = PNSynapse(gPN))
    PNs.add_edge(p[4], p[5], synapse = PNSynapse(gPN))
    PNs.add_edge(p[5], p[3], synapse = PNSynapse(gPN))
    PNs.add_edge(p[5], p[2], synapse = PNSynapse(gPN))

    #connect LNs and PNs together
    AL = nx.compose(LNs, PNs)
    nLN, nPN = list(LNs.nodes()), list(PNs.nodes())

    AL.add_edge(nLN[0], nPN[0], synapse = LNSynapse(gLNPN))
    AL.add_edge(nLN[0], nPN[1], synapse = LNSynapse(gLNPN))
    AL.add_edge(nLN[0], nPN[2], synapse = LNSynapse(gLNPN))
    AL.add_edge(nLN[1], nPN[3], synapse = LNSynapse(gLNPN))
    AL.add_edge(nLN[1], nPN[4], synapse = LNSynapse(gLNPN))
    AL.add_edge(nLN[1], nPN[5], synapse = LNSynapse(gLNPN))

    AL.add_edge(nPN[1], nLN[0], synapse = PNSynapse(gPNLN))
    AL.add_edge(nPN[3], nLN[0], synapse = PNSynapse(gPNLN))
    AL.add_edge(nPN[1], nLN[1], synapse = PNSynapse(gPNLN))
    AL.add_edge(nPN[3], nLN[1], synapse = PNSynapse(gPNLN))
    #AL.add_edge(nPN[4], nLN[1], synapse = PNSynapse(gPNLN))

    AL.layers = LNs.layers + PNs.layers
    AL.labels = LNs.labels + PNs.labels

    return AL

#Creates AL from the 2001 Bazhenov paper
def create_AL_man(LNClass, PNClass, LNSynapse, PNSynapse, gLN=400.0, gLNPN=800.0, gPN=350.0, gPNLN=300.0):
    LNs = get_single_layer(LNClass, 2)
    PNs = get_single_layer(PNClass, 6)

    AL = manual_connect(LNs, PNs, LNSynapse, PNSynapse, gLN, gLNPN, gPN, gPNLN)
    return AL


#Create_AL creates AL with random connections with prob 0.5
def create_AL(LNClass, PNClass, LNSynapse, PNSynapse, neuron_nums, gLN = 400.0, gLNPN = 800, gPN = 350, gPNLN = 300):
    num_layer = len(neuron_nums)
    LNs = get_single_layer(LNClass, neuron_nums[0])
    PNs = get_single_layer(PNClass, neuron_nums[1])

    connect_prob_LN = 0.5
    connect_prob_PN = 0.5
    if gLN >= 0:
        connect_layer(LNs, LNSynapse, connect_prob_LN, gLN)
    if gPN >= 0:
        connect_layer(PNs, PNSynapse, connect_prob_PN, gPN)

    inter_connect_prob_LN = 0.5
    inter_connect_prob_PN = 0.5
    AL = interconnect(LNs, PNs, LNSynapse, PNSynapse,
                      inter_connect_prob_LN, inter_connect_prob_PN,
                      gLNPN, gPNLN)
    return AL
