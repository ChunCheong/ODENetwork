# main.py
import numpy as np
import networks
import neuron_models as nm
import experiments
import lab_manager
# from importlib import reload  # Python 3.4+ only.
# reload(networks)
# reload(experiments)
# reload(lab_manager)

# Step 1: Pick a network and visualize it
neuron_nums = [2,1]
net = networks.get_multilayer_fc(
    nm.HHNeuronWithCa, nm.PlasticNMDASynapse, neuron_nums)
networks.draw_layered_digraph(net)

# step 2: design an experiment. (Fixing input currents really)
experiments.delay_pulses_on_layer_0_and_1(net)

# step 3: ask our lab manager to set up the lab for the experiment.
f, initial_conditions = lab_manager.set_up_lab(net)

# step 4: run the lab and gather data
time_sampled_range = np.arange(0., 100, 0.1)
data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)

# Time to witness some magic
lab_manager.sample_plot(time_sampled_range, data, net)
