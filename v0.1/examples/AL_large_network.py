# begin boiler plate for compatibility
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import sys
# This is a janky solution to import the modules. We will need to go back
# and make this a proper package
sys.path.append('..')

import lab_manager as lm # have to combine two lab_manager later

import scipy as sp
import numpy as np
from jitcode import jitcode, y, t
try:
    import symengine as sym_backend
except:
    import sympy as sym_backend
import networkx as nx
import random
import networks as net
import neuron_models as nm
import experiments as ex
from itertools import chain

#First number is #LNs, second is #PNs
neuron_nums = [30, 90]
#Create_AL creates AL with random connections with prob 0.5
AL = net.create_AL(nm.LN, nm.PN_2, nm.Synapse_gaba_LN_with_slow, nm.Synapse_nAch_PN_2, \
neuron_nums, gLN = 100.0, gLNPN = 400.0, gPN = 0.0, gPNLN = 600.0)
#Set up the experiment
num_layers = 2

# Roughly the max current each neuron is injected with. This block of code makes it
# so that random neurons are injected with current, and the levels are slighlty different
Iscale = 500
I_ext = []
for i in range(num_layers):
    I_ext.append(np.random.rand(neuron_nums[i]))
    I_ext[i][(np.nonzero(I_ext[i] >= 0.67))] = 1.0

    I_ext[i] = np.floor(I_ext[i])
    I_ext[i][np.nonzero(I_ext[i])] = Iscale*np.asarray(I_ext[i][np.nonzero(I_ext[i])]) #+ 0.02*np.random.randn(np.count_nonzero(I_ext[i]),))
neuron_inds = [np.nonzero(I_ext[j])[0].tolist() for j in range(num_layers)]
current_vals = [I_ext[j][np.nonzero(I_ext[j])] for j in range(num_layers)]

ex.const_current(AL, num_layers, neuron_inds, current_vals)

#set up the lab
f, initial_conditions, neuron_inds  = lm.set_up_lab(AL)

#run for specified time with dt
time_len = 3000.0
dt = 0.02
time_sampled_range = np.arange(0., time_len, dt)

data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5',compile=True)

pn_inds = np.array([n.ii for n in AL.layers[1].nodes()])
ln_inds = np.array([n.ii for n in AL.layers[0].nodes()])
inds = np.append(np.asarray(ln_inds),np.asarray(pn_inds))

sol = np.transpose(data)
#np.save('large_network.npy',sol[inds])


# This is code to export an adjacency matrix
#import networkx as nx
#import matplotlib.pyplot as plt
#np.savetxt('adj_mat.dat',nx.to_numpy_matrix(AL))


lm.show_random_neuron_in_layer(time_sampled_range,data,AL,0,2)
lm.show_random_neuron_in_layer(time_sampled_range,data,AL,1,6)
lm.plot_LFP(time_sampled_range, data, AL, layer_pn = 1)
