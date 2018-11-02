# main.py
from importlib import reload  # Python 3.4+ only.
from jitcode import jitcode, y, t # this "y" will now allow symbolic tracking
import symengine
import numpy as np
import networks #; reload(networks)
import neuron_models as nm #; reload(nm)
import electrodes
import matplotlib.pyplot as plt

# Step 1: Pick a network
num_neurons_layer_1, num_neurons_layer_2 = 2, 3
net = networks.HHSTDPFeedForwardFC2Layer(
    num_neurons_layer_1, num_neurons_layer_2)

# step 2: define externel/injected currents, if needed
i_max = 50. #5. # (some unit)
t0 = 50. # ms
dt = 10.
w = 1. #ms
for neuron in net.layer_1:
    neuron.i_inj = i_max*electrodes.unit_pulse(t,t0,w) # the jitcode t
for neuron in net.layer_2:
    neuron.i_inj = i_max*electrodes.unit_pulse(t,t0+dt,w)

# step 3: have the ODEs ready
def f():
    ii = 0 # integration index
    adja_list = net.adja_list # the list of lists of pre-synaptic neurons
    synapse_lists = net.edges_list # the list of lists of pre-synapses
    # step 3a: fix the integration indices sequencially
    neurons = net.vertexs
    for (n, pos_neuron) in enumerate(neurons):
        pos_neuron.fix_neuron_index(n)
        if pos_neuron.DIM: # same as if pos_neuron.DIM > 0
            pos_neuron.fix_integration_index(ii)
            ii += pos_neuron.DIM
        pre_synapses = synapse_lists[n]
        for synapse in pre_synapses:
            if synapse.DIM:
                synapse.fix_integration_index(ii)
                ii += synapse.DIM
    # step 3b: must yield the derivatives in the exact same order in step 3a
    for (n, pos_neuron) in enumerate(neurons):
        pre_neurons = [neurons[m] for m in adja_list[n]]
        pre_synapses = synapse_lists[n]
        yield from pos_neuron.dydt(pre_synapses, pre_neurons)
        for (j, synapse) in enumerate(pre_synapses):
            yield from synapse.dydt(pre_neurons[j], pos_neuron)

# step 4: impose initial conditions. Must follow the same order in step 3a.
initial_conditions = []
neurons = net.vertexs # the list of all neruons
synapse_lists = net.edges_list # the list of lists of pre-synapses

for (n, pos_neuron) in enumerate(neurons):
    pre_synapses = synapse_lists[n]
    if pos_neuron.DIM:
        initial_conditions += pos_neuron.get_initial_condition()
    for synapse in pre_synapses:
        if synapse.DIM:
            initial_conditions += synapse.get_initial_condition()
initial_conditions = np.array(initial_conditions)

# step 5: pass it to jitcode or whatever we are using. voila!
dim_total = len(initial_conditions)
ODE = jitcode(f, n=dim_total)
ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
ODE.set_integrator('dopri5')# ,nsteps=10000000)
ODE.set_initial_value(initial_conditions, 0.0)
T_sampled_range = np.arange(0., 100, 0.1)
data = np.vstack(ODE.integrate(T) for T in T_sampled_range)

# Time to witness some magic
neuron_1 = net.layer_1[0] # just pick one neuron from each layer
neuron_2 = net.layer_2[0]
syn = net.edges_list[neuron_2.ni][0]
THETA_D = syn.THETA_D
THETA_P = syn.THETA_P
for (n, neuron) in enumerate([neuron_1, neuron_2]):
    ii = neuron.ii
    #dim = neuron.DIM
    v_m = data[:,ii]
    ca = data[:,ii+6]
    fig, axes = plt.subplots(2,1,sharex=True)
    axes[0].plot(T_sampled_range, v_m, label="V_m")
    axes[0].set_ylabel("V_m [mV]")
    axes[0].legend()
    axes[1].plot(T_sampled_range, ca, label="[Ca]")
    axes[1].set_ylabel("[a.u.]")
    axes[1].axhline(THETA_D, color="orange", label="theta_d")
    axes[1].axhline(THETA_P, color="green", label="theta_p")
    axes[1].legend()
    axes[-1].set_xlabel("time [ms]")
    plt.suptitle("Neuron {}".format(n))

syn = net.edges_list[neuron_2.ni][0]
ii = syn.ii
plt.figure()
syn_weight = data[:,ii]
plt.plot(T_sampled_range, syn_weight, label="w_ij")
plt.xlabel("time [ms]")
plt.legend()

## begin debug
# for dydt in f():
#     print(dydt)
