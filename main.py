# main.py
from importlib import reload  # Python 3.4+ only.
from jitcode import t # symbolic time varibale, useful for defining currents
import numpy as np
import networks #; reload(networks)
import electrodes
import lab_manager#; reload(lab_manager)

# Step 1: Pick a network
num_neurons_layer_1, num_neurons_layer_2 = 2, 3
net = networks.HHSTDPFeedForwardFC2Layer(
    num_neurons_layer_1, num_neurons_layer_2)

# step 2: Define externel/injected currents, if needed
i_max = 50. #5. # (some unit)
t0 = 50. # ms
dt = 10.
w = 1. #ms
for neuron in net.layer_1:
    neuron.i_inj = i_max*electrodes.unit_pulse(t,t0,w) # the jitcode t
for neuron in net.layer_2:
    neuron.i_inj = i_max*electrodes.unit_pulse(t,t0+dt,w)

# step 3: ask our lab manager to set up the lab.
f, initial_conditions = lab_manager.set_up_lab(net)

# step 4: run the lab and gather data
time_sampled_range = np.arange(0., 100, 0.1)
data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)

# Time to witness some magic
lab_manager.sample_plot(time_sampled_range, data, net)
