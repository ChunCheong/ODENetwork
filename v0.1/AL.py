# begin boiler plate for compatibility
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import sys
if sys.version_info.major > 2:
    xrange = range
elif sys.version_info.major == 2:
    pass
# end boiler plate for compatibility

import lab_manager as lm # have to combine two lab_manager later

import scipy as sp
import pylab as plt
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

# import importlib # only for python 3.4+
# importlib.reload(lm)
#importlib.reload(ex)

plt.style.use('ggplot')

#First number is #LNs, second is #PNs
neuron_nums = [2, 6] # 2 LNs and 6PNs
#Create_AL creates AL with random connections with prob 0.5
AL = net.create_AL(nm.LN, nm.PN, nm.Synapse_gaba_LN, nm.Synapse_nAch_PN, neuron_nums)

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

lm.show_all_neuron_in_layer(time_sampled_range, data, AL, 0)
# lm.plot_LFP(time_sampled_range, data, AL, layer_pn = 1)
