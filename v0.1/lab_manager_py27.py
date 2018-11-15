"""
Lab manager functions that work with python 2.7
"""
# begin boiler plate for compatibility
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import sys
if sys.version_info.major > 2:
    xrange = range
elif sys.version_info.major == 2:
    pass
# end boiler plate for compatibility

import numpy as np
from jitcode import jitcode, y, t # this "y" will now allow symbolic tracking
import networks #; reload(networks)
import electrodes#; reload(electrodes)
import neuron_models as nm
import matplotlib.pyplot as plt


def set_up_lab(net):
    neurons = net.nodes()
    neuron_inds = []
    # step 3a: fix the integration indices sequencially
    ii = 0 # integration index
    for (n, pos_neuron) in enumerate(neurons):
        pos_neuron.set_neuron_index(n) # maybe it will be usefull?
        neuron_inds.append(ii)
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
    # Have the ODEs ready construct a generator for that
    def f():
        # adja_list = net.adja_list # the list of lists of pre-synaptic neurons
        # synapse_lists = net.edges_list # the list of lists of pre-synapses
        # step 3b: must yield the derivatives in the exact same order in step 3a
        for (n, pos_neuron) in enumerate(neurons):
            pre_neurons = [neuron for neuron in net.predecessors(pos_neuron)] # can try replace [] -> ()
            pre_synapses = [
                net[pre_neuron][pos_neuron]["synapse"]
                for pre_neuron in net.predecessors(pos_neuron)]
            for eq in pos_neuron.dydt(pre_synapses, pre_neurons):
                yield eq
            for pre_neuron in net.predecessors(pos_neuron):
                synapse = net[pre_neuron][pos_neuron]["synapse"]
                for eq in synapse.dydt(pre_neuron, pos_neuron):
                    yield eq
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
    return f, initial_conditions, neuron_inds

def run_lab(f, initial_conditions, time_sampled_range, integrator='dopri5'):
    dim_total = len(initial_conditions)
    ODE = jitcode(f, n=dim_total)
    ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
    ODE.set_integrator(integrator)# ,nsteps=10000000)
    ODE.set_initial_value(initial_conditions, 0.0)
    data = np.vstack(ODE.integrate(T) for T in time_sampled_range)
    return data





"""
show_layer(time_sampled_range, data, net, layer_idx):
"""
def show_all_neuron_in_layer(time_sampled_range, data, net, layer_idx):
    neurons = net.layers[layer_idx].nodes()
    for (n, neuron) in enumerate(neurons):
        # fig = plt.figure()
        ii = neuron.ii
        v_m = data[:,ii]
        I = list(map(neuron.i_inj, time_sampled_range))

        fig, axes = plt.subplots(2,1,sharex=True)
        axes[1].plot(time_sampled_range, I, label="i_inj")
        axes[1].set_ylabel("I (pA)")
        axes[1].legend()
        axes[0].plot(time_sampled_range, v_m, label="V_m")
        axes[0].set_ylabel("V_m [mV]")
        axes[0].legend()
        axes[-1].set_xlabel("time [ms]")
        plt.suptitle("Neuron {} in layer {}".format(neuron.ni, layer_idx))
    plt.show()

"""
Plot the local field potential
"""
def plot_LFP(time_sampled_range, data, net, layer_pn = 1):
    t = time_sampled_range
    fig = plt.figure(figsize = (8,5))
    plt.title('Local Field Potential')
    plt.ylabel('LFP (mV)')
    plt.xlabel('time (ms)')
    inds = np.array([n.ii for n in net.layers[layer_pn].nodes()])
    sol = np.transpose(data)
    plt.plot(t, np.mean(sol[inds], axis = 0))
    plt.show()
    #fig.savefig('LFP62.pdf', bbox_inches='tight')
