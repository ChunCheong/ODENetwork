# main.py
import numpy as np
import networks
import neuron_models as nm
import experiments
import lab_manager
#from importlib import reload  # Python 3.4+ only.
# reload(networks)
# reload(experiments)
#reload(lab_manager)
# Step 1: Pick a network and visualize it
neuron_nums = [2,1] # number of neurons in each layer
net = networks.get_multilayer_fc(
    nm.HHNeuronWithCa, nm.PlasticNMDASynapseWithCa, neuron_nums)
networks.draw_layered_digraph(net)

# step 2: design an experiment. (Fixing input currents really)
#experiments.delay_pulses_on_layer_0_and_1(net)
i_max=50.
num_sniffs=2
time_per_sniff=100.
total_time = num_sniffs*time_per_sniff
base_rate = 0.1
experiments.feed_gaussian_rate_poisson_spikes(
    net, base_rate, i_max=i_max, num_sniffs=num_sniffs,
    time_per_sniff=time_per_sniff)

# step 3: ask our lab manager to set up the lab for the experiment.
f, initial_conditions = lab_manager.set_up_lab(net)

# step 4: run the lab and gather data
time_sampled_range = np.arange(0., total_time, 0.1)
data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)

# Time to witness some magic
#lab_manager.sample_plot(time_sampled_range, data, net)
for layer_idx in range(len(net.layers)):
    lab_manager.show_all_neuron_in_layer(
        time_sampled_range, data, net, layer_idx)
    lab_manager.show_all_synaspe_onto_layer(
        time_sampled_range, data, net, layer_idx)
