import sys
import lab_manager as lm

import scipy as sp
import numpy as np
from jitcode import jitcode, y, t
import networkx as nx
import random
import networks as net
import neuron_models as nm
import experiments as ex
from itertools import chain

#First number is #LNs, second is #PNs
neuron_nums = [30, 90]
#Create_AL creates AL with random connections with prob 0.5, -1 conductivity -> no connection
AL = net.create_AL(nm.LN, nm.PN_2, nm.Synapse_gaba_LN_with_slow, nm.Synapse_nAch_PN_2, \
neuron_nums, gLN = 100.0, gLNPN = 400.0, gPN = -1.0, gPNLN = 600.0)
#Set up the experiment
num_layers = 2


num_odors = int(input('Enter number of odours: '))
num_per_od = int(input('Enter # different concentrations per od: '))

for number in range(num_odors):
#Each odour is encoded by injected current in different subsets of neurons
	curr_neurons = [np.random.rand(neuron_nums[0]), np.random.rand(neuron_nums[1])]
	for j in range(num_per_od):
		print(j)

		p = 0.33 #proportion of neurons with injected current

		# Roughly the max current each neuron is injected with.
		#present each odor at different concentrations
		Iscale = j*150 + 150
		I_ext = []
		for i in range(num_layers):
			#choose the neurons to get current
		    I_ext.append(curr_neurons[i])
		    I_ext[i][(np.nonzero(I_ext[i] >= (1-p)))] = 1.0

		    I_ext[i] = np.floor(I_ext[i])
		    I_ext[i][np.nonzero(I_ext[i])] = Iscale*np.asarray(I_ext[i][np.nonzero(I_ext[i])])
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
	#	od.append(sol)
		np.save('Data/AL_3090_' + str(Iscale)+ '_'+str(number), sol[pn_inds])

#np.save('large_network.npy',sol[inds])


# This is code to export an adjacency matrix
#import networkx as nx
#import matplotlib.pyplot as plt
#np.savetxt('adj_mat.dat',nx.to_numpy_matrix(AL))


# lm.show_random_neuron_in_layer(time_sampled_range,data,AL,0,2)
# lm.show_random_neuron_in_layer(time_sampled_range,data,AL,1,6)
# lm.plot_LFP(time_sampled_range, data, AL, layer_pn = 1)
