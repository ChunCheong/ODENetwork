# jitcode_oop_example.py
from jitcode import jitcode, y # this "y" will now allow symbolic traccouple_consting
import symengine
import numpy as np
import matplotlib.pyplot as plt

NUM_NEURON = 20
NUM_DIM_NEURON = 1 # the dimension (no. of variables) of a single neuron.
NUM_SYN = 0 # For simplicity in the example we have no dynamical synapse.
NUM_DIM_SYN = 0 # the dimension (no. of variables) of a single synapse.
NUM_DIM_TOT = NUM_NEURON*NUM_DIM_NEURON + NUM_SYN*NUM_DIM_SYN
# kuramoto model
int_freqs = np.random.uniform(0.8,1.0,NUM_NEURON) # intrinsic frequency of each oscillator
couple_const = 0.2 # coupling strength
# Get an adjacency matrix of some network
A = np.random.rand(NUM_NEURON,NUM_NEURON); A = A>0.5
# and equivalently an adjaceny list
A_list = [np.where(A[i,:])[0] for i in range(NUM_NEURON)]
#print(A_list[0])
"""
I chose to use the simnplest kind of data structure, namly listing all
variables one by one: x[0], v[0], z[0], x[1], …, x[N], v[N], z[N]
Hence the the function n2i(n) maps the neuron index 0<=n< NUM_NEURON to the
integration variable index 0<=i<NUM_DIM_TOT, of the first dimension of that
neuron.
"""
def n2i(n):
    return n*NUM_DIM_NEURON

"""A similar function for synapse"""
def s2i(s):
    pass

class Neuron:
    def __init__(self,n,para):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        i = n2i(n) #integration variable index
        self.mem_pot = y(i) # the jitcode y
        # self.m = y(i+1)
        # ... and stuff like that
        self.int_freq = para[0]
        self.dim = NUM_DIM_NEURON # can vary across different types of neuron

    def dydt(self, pre_syn_neurons): # a list of pre-synaptic neurons
        # define how neurons are coupled here
        v_i = self.mem_pot
        coupling_sum = sum(
            symengine.sin(n.mem_pot-v_i) for n in pre_syn_neurons)
        coupling_term = couple_const * coupling_sum
        yield self.int_freq + coupling_term

class Syanpse:
    pass

# As a example, we create an instance of a Neuron, called n0.
# para0 = [int_freqs[0]]
# n0 = Neuron(0,para0)
# print(n0.mem_pot)
# print(n0.int_freq)
# print(n0.dim)
# pre_syn_neurons = [] # a trivial example
# for dydt in n0.dydt(pre_syn_neurons):
#     print(dydt)
# A_list[0]

# To generate a ODEs, we will create instances of neurons and hook them up.
# step 1: create Neuron (and Synapse) instances
neurons = [Neuron(n,[int_freqs[n]]) for n in range(NUM_NEURON)]
# step 2: define our ODEs
def f():
    for (n, neuron) in enumerate(neurons):
        pre_syn_neurons = [neurons[m] for m in A_list[n]]
        yield from neuron.dydt(pre_syn_neurons)

# One can check f() defined this way is the same as the "hard-coded" one.
# #print(f()) <generator object f at 0x0000015B0C9CABA0>
# for dydt in f():
#    print(dydt)

initial_state = 2*np.random.random(NUM_DIM_TOT) - 1

ODE = jitcode(f, n=NUM_DIM_TOT)
ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
ODE.set_integrator('dopri5')
ODE.set_initial_value(initial_state,0.0)

data = np.vstack(ODE.integrate(T) for T in np.arange(0., 10., 0.1))
data.shape

plt.figure()
for i in range(NUM_DIM_TOT):
    plt.plot(np.sin(data[:,i]))
plt.show()
