# main.py
from importlib import reload  # Python 3.4+ only.
import numpy as np
import networks ; reload(networks)
import experiments; reload(experiments)
import lab_manager ; reload(lab_manager)

# Step 1: Pick a network
num_neurons_layer_1, num_neurons_layer_2 = 2, 3
net = networks.HHSTDPFeedForwardFC2Layer(
    num_neurons_layer_1, num_neurons_layer_2)

# step 2: design an experiment. (Fixing input currents really)
experiments.delay_pulses_on_layer_1_and_2(net)

# step 3: ask our lab manager to set up the lab for the experiment.
f, initial_conditions = lab_manager.set_up_lab(net)

# step 4: run the lab and gather data
time_sampled_range = np.arange(0., 100, 0.1)
data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)

# Time to witness some magic
lab_manager.sample_plot(time_sampled_range, data, net)
