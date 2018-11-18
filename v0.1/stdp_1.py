#stdp_1.py
import numpy as np
import matplotlib.pyplot as plt

def heaviside(t):
    K=1e3
    return 1./(1.+ np.exp(-np.clip(t*K,-10,10)))

def c_syn(t, TAU_RHO=0.5):
    t = np.clip(t,0,50)
    MAX_SYN_CA = 1/TAU_RHO**(1/(TAU_RHO-1)) - 1/TAU_RHO**(TAU_RHO/(TAU_RHO-1))
    return 1/MAX_SYN_CA*(np.exp(-t/TAU_RHO) - np.exp(-t))

def c_pos(t):
    t_ = np.clip(t,0,50)
    return np.exp(-t_)*heaviside(t)

def time_above(c, theta):
    return sum(c>theta)

def dw(c, THETA_D, THETA_P, GAMMA_D):
    time_pot = time_above(c, THETA_P)
    time_dep = time_above(c, THETA_D)
    dw = time_pot - time_dep*GAMMA_D
    return dw

# CHAPTER 1: Simplified model

# In the following, I am working in unit such that
# TAU_CA = 1
# C_POS_MAX = 1
def get_simplified_stdp_profile(
    max_c_syn=0.39,
    THETA_D=0.4,
    THETA_P=0.85,
    TAU_RHO=3.):
    t_range = np.arange(0.,20,0.01)
    delay_range = np.arange(-10,10,0.1)
    stdp_curve = np.zeros_like(delay_range)
    #max_c_syn, THETA_D, THETA_P, TAU_RHO = 0.39, 0.4, 0.85,3
    GAMMA_D = 1*np.log(THETA_P)/np.log(THETA_D)

    for i, delay in enumerate(delay_range):
        t0 = 5
        cp = c_pos(t_range-t0)
        cs = max_c_syn*c_syn(t_range-t0+delay,TAU_RHO=TAU_RHO)
        c = cp + cs
        # plt.figure()
        # plt.plot(t_range, cp, label="C_pos")
        # plt.plot(t_range, cs, label="C_syn")
        # plt.plot(t_range, c, label="C_tot")
        stdp_curve[i] = dw(c, THETA_D, THETA_P, GAMMA_D)

    plt.figure()
    plt.plot(delay_range,stdp_curve,marker="o")
    plt.grid()

# A centered one
get_simplified_stdp_profile(
    max_c_syn=0.39, THETA_D=0.4, THETA_P=0.85, TAU_RHO=3.)
# A slightly shifted one, can be more useful in some case
# expecially when our synapse has no time delay
get_simplified_stdp_profile(
    max_c_syn=0.39, THETA_D=0.4, THETA_P=0.85, TAU_RHO=1.5)
#plt.close("all")

# Now we are playing with the parameter space a bit,
# eventually one might want fit the model with experimental data

# CHAPTER 2: Full model
# Now play with full model
import neuron_models as nm
import lab_manager
import networks
import experiments
import numpy as np

import importlib
importlib.reload(nm)
# importlib.reload(lab_manager)
# importlib.reload(networks)
# importlib.reload(experiments)

def get_steady_state(net):
    lab_manager.reset_lab(net)
    f, initial_conditions, _ = lab_manager.set_up_lab(net)
    total_time = 100.
    # step 4: run the lab and gather data
    time_sampled_range = [total_time]
    data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)
    return data[-1,:]

def get_InsulatedNeuron(Neuron):
    class InsulatedNeuron(Neuron):
        def __init__(self):
            super().__init__()
            self.COND_SYN = 0.
    return InsulatedNeuron

def stdp_expt(net, dt, ics): # please put a 1-1 network
    def sigmoid(x):
        return 1./(1.+ np.exp(-x))
    lab_manager.reset_lab(net)
    dts = [dt]
    experiments.delay_pulses_on_layer_0_and_1(net, [50., 50+dt])
    # step 3: ask our lab manager to set up the lab for the experiment.
    f, _, _ = lab_manager.set_up_lab(net)
    ii = 0
    for edge in net.edges:
        n1, n2 = edge
        syn = net[n1][n2]["synapse"]
        ii = syn.ii
    # step 4: run the lab and gather data
    total_time = 100.
    time_sampled_range = np.arange(0., total_time, 0.1)
    data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)
    #print(ii)
    dw = sigmoid(data[-1,ii]) - sigmoid(data[0,ii])
    return dw, data

def get_stdp_profile(Neuron,Synapse,initial_conditions):
    Neuron = get_InsulatedNeuron(Neuron)
    # run stdp experiment
    stdp_time_range = np.arange(-40,40,2)
    stdp_data = np.empty_like(stdp_time_range)
    for i,dt in enumerate(stdp_time_range):
        stdp_data[i], _ = stdp_expt(net, dt, initial_conditions)

    plt.figure()
    plt.plot(stdp_time_range, stdp_data, marker="o")
    plt.grid()

# verify
get_stdp_profile(Neuron,Synapse,initial_conditions)

# A experiment with pulse train
Neuron, Synapse = nm.HHNeuronWithCaJL, nm.PlasticNMDASynapseWithCaJL
neuron_nums = [1,1]
net = networks.get_multilayer_fc(Neuron, Synapse, neuron_nums)
total_time = 1000.
time_sampled_range = np.arange(0., total_time, 0.1)
f, _, _ = lab_manager.set_up_lab(net)
t0s = np.arange(0,1000,50)
experiments.pulse_train_on_layer(net, 0, t0s, i_max=50.)
data = lab_manager.run_lab(f, initial_conditions, time_sampled_range)
lab_manager.show_all_neuron_in_layer(time_sampled_range, data, net, 0)
lab_manager.show_all_neuron_in_layer(time_sampled_range, data, net, 1)
lab_manager.show_all_synaspe_onto_layer(time_sampled_range, data, net, 1)
