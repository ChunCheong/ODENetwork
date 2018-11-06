# sratchpad.py

import symengine

def foo(x):
    return 1./(1.+ symengine.exp(-x))

def sym2num(f):
    x = symengine.symbols('x')
    expr = f(x)
    return symengine.Lambdify(x, expr)

foo(0.5)

from importlib import reload  # Python 3.4+ only.
import numpy as np
import networks #; reload(networks)
import experiments #; reload(lab_manager)
import lab_manager #; reload(lab_manager)

# Step 1: Pick a network
num_neurons_layer_1, num_neurons_layer_2 = 2, 3
net = networks.HHSTDPFeedForwardFC2Layer(
    num_neurons_layer_1, num_neurons_layer_2)

# step 2: design an experiment. (Fixing input currents really)
import matplotlib.pyplot as plt
experiments.delay_pulses_on_layer_1_and_2(net)
from jitcode import t
n0 = net.vertexs[0]
expr = n0.i_inj
from jitcode import t
t = 0. #symengine.symbols('t')
f = symengine.Lambdify(t, expr)
plt.plot(f(np.linspace(0.,100.)))
sym2num(n0.i_inj)
# step 3: ask our lab manager to set up the lab for the experiment.
f, initial_conditions = lab_manager.set_up_lab(net)

# step 4: run the lab and gather data
time_sampled_range = np.arange(0., 100, 0.1)
data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)

1
list1 = [1,2,3]

class A:
    def __init__(self, stuff):
        self.stuff = stuff
a,b,c = 1,2,3
x,y,z = 11,22,33
stuff1 = [a,b,c]
stuff2 = [z,y,z]
# when create class instance, attributes are assigned through passing by reference
a1 = A(stuff1)
a1.stuff[0] += 1
a1.stuff
stuff1
print(stuff1)
a2 = A(stuff1)
a2.stuff
a2.stuff[0] = 99
a2.stuff
a1.stuff
# conlcusion: since a1 and a2 share the same underlying data, changing
# will also change the others
stuffa1_copy = a1.stuff #does not create copy
stuffa1_copy[1] = 99
a1.stuff
a2.stuff
stuff3_newcopy = a1.stuff+a2.stuff # creates a new object
print(stuff3_newcopy)
stuff3_newcopy[0] = -99
print(stuff3_newcopy)
print(a1.stuff)

a1.stuff.append(a2.stuff) #nope
a1.stuff

a
print(stuff3)



a3 = A(a1.stuff+a2.stuff)
a3.stuff

a3.stuff[0] = -99
a3.stuff

a,b,c = [1],[2],[3]
x,y,z = [11],[22],[33]
stuff1 = [a,b,c]
stuff2 = [x,y,z]
#stuff3 = [s for s in stuff1]
# import copy
# stuff3 = copy.copy(stuff1)
# stuff3.extend(stuff2)
# stuff3[0]
stuff3 = stuff1 + stuff2 # stuff3 share the same underlying objects as stuff1 and stuff2
id(stuff3[0])
id(stuff1[0])
id(stuff3[3])
id(stuff2[0])
stuff3[3][0] += 1
id(stuff3[3])
stuff3[3]

# another experiment
a,b,c = [1],[2],[3]
x,y,z = [11],[22],[33]
stuff1 = [a,b,c]
stuff2 = [x,y,z]
stuff3 = stuff1 + stuff2 # stuff3 share the same underlying objects as stuff1 and stuff2
a1 = A(stuff1); a2 = A(stuff2)
a3 = A(stuff3)
id(a3.stuff[0]) == id(a1.stuff[0]) # voila!
id(stuff1[0])
# another lab
stuff3 = stuff1 + stuff1 # what will happen?
id(stuff3[0]) == id(stuff3[3]) # interesting
stuff3[0][0] +=1
print(stuff3)
d = [4]
id(d)
stuff1.append(d)
id(stuff1[3])

class A:
    def __init__(self, stuff, subgraphs=None):
        self.stuff = stuff
        self.subgraphs = subgraphs
        self.set_subgraphs()
    def set_subgraphs(self):
        self.subgraphs = [self]

a1 = A(stuff1)

a1.subgraphs

from importlib import reload  # Python 3.4+ only.
import numpy as np
import networks ; reload(networks)
import neuron_models as nm; reload(nm)

import experiments #; reload(lab_manager)
import lab_manager #; reload(lab_manager)

# Step 1: Pick a network
num_neurons_layer_1, num_neurons_layer_2 = 2, 3
net1 = networks.HHSTDPFeedForwardFC2Layer(
    num_neurons_layer_1, num_neurons_layer_2)
net2 = networks.HHSTDPFeedForwardFC2Layer(
    num_neurons_layer_1, num_neurons_layer_2)
net1.vertexs[0].ni
net2.vertexs[3].ni
import networks ; reload(networks)
net3 = networks.fully_connect(net1, net2, nm.StaticSynapse)
len(net3.vertexs)
net3.adja_list



import networkx as nx

G=nx.DiGraph()

G.add_nodes_from(net1.vertexs)
net1.vertexs
dir(G)
G.nodes()
G.add_edge(net1.vertexs[0], net1.vertexs[1],synapse=nm.PlasticNMDASynapse)
G[net1.vertexs[0]][net1.vertexs[1]]
G.successors(net1.vertexs[1])
G.has_edge(net1.vertexs[0], net1.vertexs[1])

n1, n2 = net1.vertexs[0], net1.vertexs[1]
for n in G.successors(n2):
    print(n)
for n in G.predecessors(n2):
    print(n)
import numpy as np
a,b,c,d = 1, 2,3,4

G1 = nx.DiGraph();
G1.add_nodes_from([a,b])
G2 = nx.DiGraph();
G2.add_nodes_from([c,d])
G1.add_edge(a,b,synapse=1.) # a synapses onto b
nx.draw_networkx(G1)

class Foo:
    def __init__(self, x):
        self.x = x
nodes = [Foo(i) for i in range(10)];
more_nodes = [Foo(i) for i in range(10,20)]
G1 = nx.DiGraph(); G1.add_nodes_from(nodes)
G2 = nx.DiGraph(); G2.add_nodes_from(more_nodes)
G3 = nx.compose(G1,G2)
ns = G3.nodes
for (i,n) in enumerate(G3.nodes):
    if n.x <10:
        n.x = n.x + 99
for n in G3.nodes:
    print(n.x)
from importlib import reload  # Python 3.4+ only.

import networks
import neuron_models as nm
import networkx as nx
reload(networks)
reload(nm)
sl1 = networks.get_single_layer(nm.HHNeuronWithCa, 2)
sl2 = networks.get_single_layer(nm.HHNeuronWithCa, 3)
dl = networks.fully_connect(sl1, sl2, nm.PlasticNMDASynapse)
dl.layers
dl.labels
nx.draw_networkx(sl1)
nx.draw_networkx(dl, with_labels=False)
networks.draw_layered_digraph(dl)

nx.compose(sl1, sl2)

networks.draw_layered_digraph(dl)
sl1.nodes


sl1.nodes()

net = networks.get_multilayer_fc(nm.HHNeuronWithCa, nm.PlasticNMDASynapse, [2,3,1])
networks.draw_layered_digraph(net)


for pos_neuron in sl2.nodes():
    #print(pos_neuron)
    for pre_neuron in dl.predecessors(pos_neuron):
        #print(pre_neuron)
        #print(dl[pos_neuron][pre_neuron]["synapse"])
        print(dl[pre_neuron][pos_neuron]["synapse"])
        #print(dl.edges(pos_neuron))

for pos_neuron in sl2.nodes():
    #print(pos_neuron)
    pre_synapses = (
        dl[pre_neuron][pos_neuron]["synapse"]
        for pre_neuron in dl.predecessors(pos_neuron))
    for pre_synapses in pre_synapses:
        #print(pre_neuron)
        #print(dl[pos_neuron][pre_neuron]["synapse"])
        print(pre_synapses)
        #print(dl.edges(pos_neuron))


import numpy as np
import networks
import experiments
import lab_manager
from importlib import reload  # Python 3.4+ only.
reload(networks)
reload(experiments)
reload(lab_manager)

net = networks.get_multilayer_fc(
    nm.HHNeuronWithCa, nm.PlasticNMDASynapse, [2,1])
networks.draw_layered_digraph(net)

f, initial_conditions = lab_manager.set_up_lab(net)
f
for dydt in f():
    print(dydt)
f()
