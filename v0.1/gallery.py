# gallery.py
import importlib
import networks; importlib.reload(networks)
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

#1200/54 = 23

rn_para = dict(num_rn_group=10, rn_group_size=3,
    RNClass=ExcitatoryNeuron, RNSynapseClass=ExcitatorySynapse)
rn_para["RNClass"]
rn = networks.get_receptor_neurons(**rn_para)
rn.labels
networks.draw_colored_layered_digraph(rn)

glo_para = dict(num_pn=3, num_ln=5,
    PNClass=ExcitatoryNeuron, LNClass=InhibitoryNeuron,
    PNSynapseClass=ExcitatorySynapse, LNSynapseClass=InhibitorySynapse)

glo = networks.get_glomeruli(**glo_para)
glo.labels
networks.draw_colored_layered_digraph(glo)

al_para = dict(num_glo=3, glo_para=glo_para)
al = networks.get_antennal_lobe(**al_para)
al.layers
networks.draw_colored_layered_digraph(al)

mb_para = dict(num_kc=10,
    KCClass=ExcitatoryNeuron, GGNClass=InhibitoryNeuron,
    KCSynapseClass=ExcitatorySynapse, GGNSynapseClass=ExcitatorySynapse)
mb = networks.get_mushroom_body(**mb_para)
mb.labels
networks.draw_colored_layered_digraph(mb)
bl_para = dict(num_bl=4,
    BLClass=InhibitoryNeuron, BLSynapseClass=InhibitorySynapse)
bl = networks.get_beta_lobe(**bl_para)
bl.labels
networks.draw_colored_layered_digraph(bl)

other_para = dict(prob_r2a=0.5, prob_a2k=0.5, prob_k2b=0.5)
net = networks.get_olfaction_net(rn_para=rn_para, al_para=al_para, mb_para=mb_para,
    bl_para=bl_para, other_para=other_para)


nx.draw_kamada_kawai(net)
net.labels
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
