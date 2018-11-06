# workbench.py
# main.py
import numpy as np
import math
import networks
import neuron_models as nm
import experiments
import lab_manager
from importlib import reload  # Python 3.4+ only.
reload(nm)
reload(networks)
reload(experiments)
reload(lab_manager)

# Step 1: Pick a network and visualize it
neuron_nums = [2,1]
net = networks.get_multilayer_fc(
    nm.HHNeuronWithCa, nm.PlasticNMDASynapse, neuron_nums)
#networks.draw_layered_digraph(net)

# step 2: design an experiment. (Fixing input currents really)
#i_max = 30.
#experiments.delay_pulses_on_layer_0_and_1(net, i_max=i_max)
# import matplotlib.pylab as plt
# plt.figure()
# plt.plot(input_signal[0].T[0], input_signal[0].T[1], marker="o", linestyle="")
# plt.plot(input_signal[1].T[0], input_signal[1].T[1], marker="x", linestyle="")
i_max=50.
num_sniffs=10
time_per_sniff=100.
total_time = num_sniffs*time_per_sniff
base_rate = 0.1
experiments.feed_gaussian_rate_poisson_spikes(
    net, base_rate, i_max=i_max, num_sniffs=num_sniffs,
    time_per_sniff=time_per_sniff)

list(net.nodes())[1].i_inj
# step 3: ask our lab manager to set up the lab for the experiment.
f, initial_conditions = lab_manager.set_up_lab(net)
dir(list(net.nodes())[0])
# step 4: run the lab and gather data
time_sampled_range = np.arange(0., total_time, 0.1)
data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)

# Time to witness some magic
layer_idx = 0
#lab_manager.show_layer(time_sampled_range, data, net, layer_idx)
#lab_manager.plt.close("all")
for layer_idx in range(len(net.layers)):
    lab_manager.show_all_neuron_in_layer(time_sampled_range, data, net, layer_idx)
    lab_manager.show_all_synaspe_onto_layer(time_sampled_range, data, net, layer_idx)

#lab_manager.sample_plot(time_sampled_range, data, net)
#lab_manager.plt.close("all")



# import matplotlib.pyplot as plt
# mean = [0, 0]
# cov = [[1, 0], [0, 100]]
# x, y = np.random.multivariate_normal(mean, cov, 5000).T
# plt.plot(x, y, 'x')
