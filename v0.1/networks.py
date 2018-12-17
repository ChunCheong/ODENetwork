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
import itertools
import copy

import neuron_models as nm
import networkx as nx



"""
LayeredDiGraph(nx.DiGraph) is the most basic class, built upon nx.DiGraph.
It can be initialized from any existing nx.DiGraph.

Example:
dg = nx.DiGraph(); dg.add_edges_from([(1, 2), (1, 3)])
ldg = LayeredDiGraph(dg)
"""
class LayeredDiGraph(nx.DiGraph):
    def __init__(self, g=None):
        super().__init__()
        if isinstance(g, nx.DiGraph):
            self.__dict__.update(g.__dict__)
        elif g is not None:
            print("Input graph type not supported, ignoring input.")
        self.layers = [self]
        self.labels = []


"""
Populate a single layer of num_neuron NeuronClass.
"""
def get_single_layer(NeuronClass, num_neuron, label="Single Layer", graph=None):
    sl = LayeredDiGraph(graph)
    # populate instances of NeuronClass
    sl.add_nodes_from([NeuronClass() for n in range(num_neuron)])
    sl.labels = [label]
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
sparsely_connect(net1, net2, SynapseClass, prob):

Similar to fully_connect(net1, net2, SynapseClass).
Return a network with the last layer of net1 sparsely connect onto the top layer
# of net2 with SynapseClass, with probability prob making edge.

Here "top" and "last" refer to the order in net*.layers.
"""
def sparsely_connect(net1, net2, SynapseClass, prob):
    net3 = nx.compose(net1, net2)
    pre_neurons, pos_neurons = net1.layers[-1].nodes(), net2.layers[0].nodes()
    for pos_neuron in pos_neurons:
        for pre_neuron in pre_neurons:
            if random.random() < prob:
                # order matters when adding edges
                net3.add_edge(pre_neuron, pos_neuron, synapse=SynapseClass())
    net3.layers = net1.layers + net2.layers
    net3.labels = net1.labels + net2.labels
    return net3

"""
fully_connect_between(net, subgraph_1, subgraph_2, SynapseClass):

subgraph_* should be subgraphs of net and share the same underlying objects.

Unlike fully_connect(), it adds synapse(s) to the network from
subgraph_1 to subgraph_2, instead of returning a new network.
i.e input net will be modified.
"""

def fully_connect_between(net, subgraph_1, subgraph_2, SynapseClass):
    pre_neurons, pos_neurons = subgraph_1.nodes(), subgraph_2.nodes()
    for pos_neuron in pos_neurons:
        for pre_neuron in pre_neurons:
            # order matters when adding edges
            net.add_edge(pre_neuron, pos_neuron, synapse=SynapseClass())

"""
sparsely_connect_between(net, subgraph_1, subgraph_2, SynapseClass):

subgraph_* should be subgraphs of net and share the same underlying objects.

Unlike sparsely_connect(), it adds synapse(s) to the network from
subgraph_1 to subgraph_2, instead of returning a new network.
i.e input net will be modified.
"""

def sparsely_connect_between(net, subgraph_1, subgraph_2, SynapseClass, prob):
    pre_neurons, pos_neurons = subgraph_1.nodes(), subgraph_2.nodes()
    for pos_neuron in pos_neurons:
        for pre_neuron in pre_neurons:
            if random.random() < prob:
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
    net3.labels = net1.labels + net2.labels
    return net3

def stack_from_list(layers):
    net = LayeredDiGraph()
    for layer in layers:
        net = stack(net, layer)
    return net

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
get_nodes_data(net, data_key, default_val=np.nan):

Return a 1-d array with data of data_key.

Example:
net = networks.get_single_layer(Neuron, 10)
vs = get_nodes_data(net, "v")
"""
def get_nodes_data(net, data_key, default_val=np.nan):
    data = np.empty(len(net))
    for (i,n) in enumerate(net.nodes):
         try:
             data[i] = n.__dict__[data_key]
         except:
             data[i] = default_val
    return data

"""
get_edges_data(net, data_key, default_val=np.nan):

Return a 2-d array with data of data_key. The first and second index would be
the pre- and pos- synaptic neuron respectively.

Example:
net = networks.get_multilayer_fc(Neuron, Synapse, [2,3])
weights = get_nodes_data(net, "weight")
"""
def get_edges_data(net, data_key, default_val=0.):
    index = dict(zip(list(net), range(len(net))))
    data = np.full((len(net), len(net)), default_val)
    edges = nx.to_edgelist(net)
    for (n1, n2, edge) in edges:
        try:
            data[index[n1], index[n2]] = edge["synapse"].__dict__[data_key]
        except:
            data[index[n1], index[n2]] = default_val
    return data


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

"""
Draw layered graph in a structured manner, with color!.
"""
def draw_colored_layered_digraph(net):
    plt.figure()
    cycol = itertools.cycle('rbgkc')
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
        nx.draw_networkx_nodes(list(layer.nodes),pos=pos,node_color=next(cycol))
    nx.draw_networkx_edges(net,pos=pos)


"""
Antenna Lobe Functions
-------------------------------------------------------------
Please refer to the write-up for more functional/architectural details.
Schemeticaly, the insect oflaction network has a 4-layer feedforward
architecture. Also it's biologists' hobbies to name things so some nomenclatures
are inevitable.

Layer 0: Receptors Neurons (RN) [input]
Layer 1: Antennal Lobe (AL) [first stage in separation]
Layer 2: Mushroom Body (MB) [second stage in separation]
Layer 3: Beta-lobe (BL) [read-out]

Apart from the interesting architecture, unsupervised learning mechanism (STDP)
was observed experimentally between Mushroom Body and Beta-lobe.

Here we outline the architecture:

In layer 0, there are num_rn_group groups of RNs, each group consists of
rn_group_size RNs. Each group only responses to particular kind of odorant
molecule, and then connect to a subset of neuron in a glomeruli in the AL.
For now we let each RN in a group to connect a single neuron in a glomeruli,
selected randomly.

In layer 1, there are num_glomeruli groups of glomerulus. Each glomeruli
consists of num_pn excitory Projection Neurons (PNs) and num_ln inhibitory Local
Neurons (LNs), each possibly be synapsed by some RN in layer 0. Within a single
glomeruli, PNs synapse onto other PNs and LNs, making each glomeruli an echo
room with signal amplified. Between glomerulus, LNs from the a glomeruli
synapse onto other glomerulus (both their PNs and LNs). This creates a complex
and rich competition between echo boxes (glomerulus) and introduces nonlinearity
to the response, which helps separate signals. Finally, PNs also synapse
onto neurons (Kenyon Cells) randomly in next layer.

In layer 2, there are num_kc of excitory Kenyon Cells (KCs) and one inhibitory
Giant Gabaergic Neuron (GGN). All Kenyon Cells synapse onto the GGN, and the GGN
also synapse onto all KCs. It again creates a complex and rich competition
between KCs. It has be observed experimentally, the spatial-temporal pattern
among KCs is very stereotyped and specific to a particular odor input. Finally,
all KCs synapse onto neurons in BL probabilistically.

In layer 3, there are num_bln inhibitory BL neuron, each receiving input from
multiple KCs. Typically, num_bln << num_kc. These BL neuron synapse onto each
others probabilistically and (again) compete with each others. STDP, an
unsupervised learning mechanism, were observed experimentally beteen KCs and BLs.
Following Diehl and Cook's (2015) observation, we expect STDP would lead to
specialization of BLs. i.e. each BLs would response very selectively to a
particular input odor.

"""

"""
get_receptor_neurons(num_rn_group, rn_group_size):

Layer 0 of the whole olfaction network.
Return a network made of num_rn_group groups of RNs, each group consists of
rn_group_size RNs, which are instances of NeuronClass.
"""
def get_receptor_neurons(*, num_rn_group, rn_group_size,
    RNClass, RNSynapseClass):
    net = LayeredDiGraph()
    for i in range(num_rn_group):
        rn_group = get_single_layer(
            RNClass, rn_group_size, label="rn_group_{}".format(i))
        net = stack(net, rn_group)
    return net

"""
get_glomeruli(num_pn, num_ln, PNClass, LNClass, P2NSynapseClass,
    prob_p2p, prob_p2l):

In each glomeruli, PNs (sparsely?) connect to LNs.
LNs (sparsely?) connect to PNs.
Not sure whether PNs are connected between each others.
"""
def get_glomeruli(*, num_pn, num_ln,
    PNClass, LNClass, PNSynapseClass, LNSynapseClass,
    prob_p2p=0., prob_p2l=0.5, prob_l2p=0.5, prob_l2l=0.5):
    #pn_graph = fast_gnp_random_graph(num_pn, prob_p2p, directed=True) # Erdos-Renyi
    #pns = get_single_layer(PNClass, num_pn, label="pn", graph=pn_graph)
    pns = get_single_layer(PNClass, num_pn, label="PNs")
    lns = get_single_layer(LNClass, num_ln, label="LNs")
    net = stack(pns, lns)
    # ignoring p2p for now, cause I don't undertsand it
    # Also depending on how glomeruli are connected, some of below
    # could be duplicated. For now possible duplicates are commented.
    #sparsely_connect_between(net, pns, pns, PNSynapseClass, prob_p2p)
    sparsely_connect_between(net, pns, lns, PNSynapseClass, prob_p2l)
    sparsely_connect_between(net, lns, pns, LNSynapseClass, prob_l2p)
    sparsely_connect_between(net, lns, lns, LNSynapseClass, prob_l2l)
    return net

"""
get_antennal_lobe(num_glo, num_pn, num_ln, PNClass, LNClass, P2NSynapseClass,
    prob_p2l=0.5, prob_l2p=0.)

The AL consists of num_glo glomerulus.
The LNs in each glomeruli connect to PNs and LNs in other glomeruli
probabilistically with prob_l2p and prob_l2l, respectively.
"""
def get_antennal_lobe(*, num_glo, glo_para,
    prob_p2p=0., prob_p2l=0.5, prob_l2p=0.5, prob_l2l=0.5):
    # first propulate glomerulus
    net = LayeredDiGraph()
    net.layers = []
    net.labels = []
    for i in range(num_glo):
        glo = get_glomeruli(**glo_para)
        layers_, labels_ = net.layers, net.labels
        net = nx.compose(net, glo)
        net.layers = layers_ + [glo]
        net.labels = labels_ + ["glo_{}".format(i)]
    # # For each glomeruli, connect LNs to all glomeruli
    LNSynapseClass = glo_para["LNSynapseClass"]
    for glo_1 in net.layers:
        for glo_2 in net.layers:
            lns_1 = glo_1.layers[1]
            pns_2 = glo_2.layers[0]
            lns_2 = glo_2.layers[1]
            sparsely_connect_between(
                net, lns_1, pns_2, LNSynapseClass, prob_l2p)
            sparsely_connect_between(
                net, lns_1, lns_2, LNSynapseClass, prob_l2l)
    return net


"""
get_mushroom_body(num_kc,
    KCClass, GGNClass, KCSynapseClass, GGNSynapseClass)
"""
def get_mushroom_body(*, num_kc,
    KCClass, GGNClass, KCSynapseClass, GGNSynapseClass):
    kcs = get_single_layer(KCClass, num_kc, label="KCs")
    ggn = get_single_layer(GGNClass, 1, label="GGN")
    net = stack(kcs, ggn)
    fully_connect_between(net, kcs, ggn, KCSynapseClass)
    fully_connect_between(net, ggn, kcs, GGNSynapseClass)
    return net

"""
get_beta_lobe(num_bl, BLClass, BLSynapseClass, prob_b2b)
"""
def get_beta_lobe(*, num_bl, BLClass, BLSynapseClass, prob_b2b=0.5):
    net = get_single_layer(BLClass, num_bl, label="BLs")
    sparsely_connect_between(net, net, net, BLSynapseClass, prob_b2b)
    return net

"""
get_olfaction_net(*, rn_para, al_para, mb_para, bl_para)
"""

def get_olfaction_net(*, rn_para, al_para, mb_para, bl_para, other_para):
    RNSynapseClass = rn_para["RNSynapseClass"]
    PNSynapseClass = al_para["glo_para"]["PNSynapseClass"]
    KCSynapseClass = mb_para["KCSynapseClass"]
    prob_r2a = other_para["prob_r2a"]
    prob_a2k = other_para["prob_a2k"]
    prob_k2b = other_para["prob_k2b"]
    rn = get_receptor_neurons(**rn_para)
    al = get_antennal_lobe(**al_para)
    mb = get_mushroom_body(**mb_para)
    bl = get_beta_lobe(**bl_para)
    layers = [rn, al, mb, bl]
    net = stack_from_list(layers)
    for rn_group in rn.layers:
        glo = next(iter(al.layers))
        sparsely_connect_between(net, rn_group, glo, RNSynapseClass, prob_r2a)
    kcs = mb.layers[0]
    for glo in al.layers:
        pns = glo.layers[0]
        sparsely_connect_between(net, pns, kcs, PNSynapseClass, prob_a2k)
    sparsely_connect_between(net, kcs, bl, KCSynapseClass, prob_k2b)
    # wrap around the labels
    net.labels = ["RN", "AL", "MB", "BL"]
    net.layers = [rn, al, mb, bl]
    return net




"""
Simplified Antenna Lobe Functions
Mostly following Bazhenov 2001
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
