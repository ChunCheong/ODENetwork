# fruitfly.py
# This script illustrates some details about the architecture of the insect
# olfaction net. Numbers iin this script are from fruit flies.
# Please refer to networks.py for details of implmentation.
# Connectivities are known up to various levels of details;
# Numbers of neurons are well-known, connectivity beteen anatomical stucture
# are known, probabilities of random connections are less clear.
# Anyway, details may change as we dig deeper over time.

import networks
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Some placeholders
class ExcitatoryNeuron:
    def __init__(self):
        pass

class ExcitatorySynapse:
    def __init__(self):
        self.weight = 1

class InhibitoryNeuron:
    def __init__(self):
        pass

class InhibitorySynapse:
    def __init__(self):
        self.weight = -1

"""
Please refer to the write-up for more functional/architectural details.
Schemeticaly, the insect oflaction network has a 4-layer feedforward
architecture.
"""

#Layer 0: Receptors Neurons (RN) [input]
rn_para = dict(num_rn_group=54, rn_group_size=23, # flies: 54 and 23
    RNClass=ExcitatoryNeuron, RNSynapseClass=ExcitatorySynapse)
rn = networks.get_receptor_neurons(**rn_para)
#networks.draw_colored_layered_digraph(rn)

#Layer 1: Antennal Lobe (AL) [first stage in separation]
glo_para = dict(num_pn=3, num_ln=30, # flies: 3 and 30
    PNClass=ExcitatoryNeuron, LNClass=InhibitoryNeuron,
    PNSynapseClass=ExcitatorySynapse, LNSynapseClass=InhibitorySynapse)

glo = networks.get_glomeruli(**glo_para)
#networks.draw_colored_layered_digraph(glo)

al_para = dict(num_glo=54, glo_para=glo_para) # flies: 54
al = networks.get_antennal_lobe(**al_para)
#networks.draw_colored_layered_digraph(al)

#Layer 2: Mushroom Body (MB) [second stage in separation]
mb_para = dict(num_kc=2500, # flies: 2500
    KCClass=ExcitatoryNeuron, GGNClass=InhibitoryNeuron,
    KCSynapseClass=ExcitatorySynapse, GGNSynapseClass=ExcitatorySynapse)
mb = networks.get_mushroom_body(**mb_para)
#networks.draw_colored_layered_digraph(mb)

#Layer 3: Beta-lobe (BL) [read-out]
bl_para = dict(num_bl=34, #flies: 34
    BLClass=InhibitoryNeuron, BLSynapseClass=InhibitorySynapse)
bl = networks.get_beta_lobe(**bl_para)
#bl.labels
#networks.draw_colored_layered_digraph(bl)

other_para = dict(prob_r2a=0.5, prob_a2k=0.5, prob_k2b=0.5)

# The whole thing
net = networks.get_olfaction_net(rn_para=rn_para, al_para=al_para,
    mb_para=mb_para, bl_para=bl_para, other_para=other_para)


#nx.draw_kamada_kawai(net)
adj_mat = networks.get_edges_data(net, "weight")
plt.imshow(adj_mat)
sizes = np.array([len(layer) for layer in net.layers])
margins = np.cumsum(sizes)
tickspos = margins - sizes/2
plt.xticks(tickspos, net.labels)
plt.yticks(tickspos, net.labels)
for margin in margins:
    plt.axhline(y=margin-0.5)
    plt.axvline(x=margin-0.5)
plt.show()

# Finally, the tunings of the recpetors are actually fairly sohpisticated, and
# are probably evolved to optimized the statisitcs of odor space.
# cf. Grabe et al., 2016, Cell Reports 16, 3401–3413 September 20, 2016
# Therefore, it might not be immediately useful for every ML task. For image
# classficiation purpose, some adapter/filter mimic recpetive field could be
# a good start. It seems Jon Larson has figured it out already. Would love to
# hear more.
