"""
lab_manager.py

It does what a lab manager should be doing. i.e
1. set_up_lab()
2. run_lab()
"""
import numpy as np
from jitcode import jitcode, y, t # this "y" will now allow symbolic tracking
import networks #; reload(networks)
import electrodes#; reload(electrodes)
import matplotlib.pyplot as plt

"""
set_up_lab(net):

Prepare all the ODEs and impose initial coonditions.
"""
def set_up_lab(net):
    neurons = net.nodes()
    # Have the ODEs ready construct a generator for that
    def f():
        ii = 0 # integration index
        # adja_list = net.adja_list # the list of lists of pre-synaptic neurons
        # synapse_lists = net.edges_list # the list of lists of pre-synapses
        # step 3a: fix the integration indices sequencially
        for (n, pos_neuron) in enumerate(neurons):
            pos_neuron.set_neuron_index(n) # maybe it will be usefull?
            if pos_neuron.DIM: # same as if pos_neuron.DIM > 0
                pos_neuron.set_integration_index(ii)
                ii += pos_neuron.DIM
                pre_synapses =  (
                    net[pre_neuron][pos_neuron]["synapse"] # order matter!
                    for pre_neuron in net.predecessors(pos_neuron))
            for pre_neuron in net.predecessors(pos_neuron):
                synapse = net[pre_neuron][pos_neuron]["synapse"]
                if synapse.DIM:
                    synapse.set_integration_index(ii)
                    ii += synapse.DIM
        # step 3b: must yield the derivatives in the exact same order in step 3a
        for (n, pos_neuron) in enumerate(neurons):
            pre_neurons = [neuron for neuron in net.predecessors(pos_neuron)] # can try replace [] -> ()
            pre_synapses = [
                net[pre_neuron][pos_neuron]["synapse"]
                for pre_neuron in net.predecessors(pos_neuron)]
            yield from pos_neuron.dydt(pre_synapses, pre_neurons)
            for pre_neuron in net.predecessors(pos_neuron):
                synapse = net[pre_neuron][pos_neuron]["synapse"]
                yield from synapse.dydt(pre_neuron, pos_neuron)
    # Impose initial conditions
    initial_conditions = []
    #neurons = net.vertexs # the list of all neruons
    #synapse_lists = net.edges_list # the list of lists of pre-synapses
    # Must follow the same order in the appearance in f()
    for (n, pos_neuron) in enumerate(neurons):
        if pos_neuron.DIM:
            initial_conditions += pos_neuron.get_initial_condition()
        pre_synapses = (
            net[pre_neuron][pos_neuron]["synapse"]
            for pre_neuron in net.predecessors(pos_neuron))
        for synapse in pre_synapses:
            if synapse.DIM:
                initial_conditions += synapse.get_initial_condition()
    initial_conditions = np.array(initial_conditions)
    return f, initial_conditions

"""
run_lab(f, initial_conditions, time_sampled_range, integrator='dopri5'):

Run the lab.
"""
def run_lab(f, initial_conditions, time_sampled_range, integrator='dopri5'):
    dim_total = len(initial_conditions)
    ODE = jitcode(f, n=dim_total)
    ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
    ODE.set_integrator(integrator)# ,nsteps=10000000)
    ODE.set_initial_value(initial_conditions, 0.0)
    data = np.vstack(ODE.integrate(T) for T in time_sampled_range)
    return data

"""
sample_plot(data, net):

Just a demo. Nothing special really.
"""
def sample_plot(time_sampled_range, data, net):
    neuron_1 = list(net.layers[0].nodes)[0] # just pick one neuron from each layer
    neuron_2 = list(net.layers[1].nodes)[0]
    syn = net[neuron_1][neuron_2]["synapse"]
    THETA_D = syn.THETA_D
    THETA_P = syn.THETA_P

    for (n, neuron) in enumerate([neuron_1, neuron_2]):
        ii = neuron.ii
        v_m = data[:,ii]
        ca = data[:,ii+6]
        i_inj = electrodes.sym2num(t, neuron.i_inj)
        i_inj = i_inj(time_sampled_range)
        fig, axes = plt.subplots(3,1,sharex=True)
        axes[0].plot(time_sampled_range, v_m, label="V_m")
        axes[0].set_ylabel("V_m [mV]")
        axes[0].legend()
        axes[1].plot(time_sampled_range, ca, label="[Ca]")
        axes[1].set_ylabel("[a.u.]")
        axes[1].axhline(THETA_D, color="orange", label="theta_d")
        axes[1].axhline(THETA_P, color="green", label="theta_p")
        axes[1].legend()
        axes[2].plot(time_sampled_range, i_inj, label="i_inj")
        axes[2].set_ylabel("[some unit]")
        axes[2].legend()
        axes[-1].set_xlabel("time [ms]")
        plt.suptitle("Neuron {}".format(n))

    ii = syn.ii
    plt.figure()
    syn_weight = data[:,ii]
    plt.plot(time_sampled_range, syn_weight, label="w_ij")
    plt.xlabel("time [ms]")
    plt.legend()
    plt.show()
    return
