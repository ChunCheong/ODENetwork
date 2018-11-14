import scipy as sp
import pylab as plt
import numpy as np
from jitcode import jitcode, y, t
import symengine
import networkx as nx
import random
import networks as net
import neuron_models as nm
import lab_manager_py27 as lm
import experiments as ex
plt.style.use('ggplot')

#First number is #LNs, second is #PNs
neuron_nums = [6,18] # 2 LNs and 6PNs
#Create_AL creates AL with random connections with prob 0.5
AL = net.create_AL(nm.LN, nm.PN, nm.Synapse_gaba_LN, nm.Synapse_nAch_PN, neuron_nums)

#Creates 6PN, 2LN network in Fig 1 Bazhenov 2001
#AL = net.create_AL_man(nm.LN, nm.PN, nm.Synapse_gaba_LN, nm.Synapse_nAch_PN)

#Set up the experiment
num_layers = 2
neuron_inds = [[0,1], [1,3]]
current_vals = [[300, 300], [300, 300]]
ex.const_current(AL, num_layers, neuron_inds, current_vals)

#set up the lab
f, initial_conditions, neuron_inds  = lm.set_up_lab(AL)

#run for specified time with dt
time_len = 200.0
dt = 0.2
time_sampled_range = np.arange(0., time_len, dt)

data = lm.run_lab(f, initial_conditions, time_sampled_range, integrator = 'dopri5')

#Plot
lm.show_all_neuron_in_layer(time_sampled_range, data, AL, 0)
lm.plot_LFP(time_sampled_range, data, AL, layer_pn = 1)