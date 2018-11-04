# jitcode_example.py
from jitcode import jitcode, y # this "y" will now allow symbolic tracking
import symengine
import numpy as np
import matplotlib.pyplot as plt

# jitcode is used to solve ivp for ODEs:
# dydy = f(y). Here we demostrate how to use it with the Kuramoto model.
# It is one of the simplest models of coupled oscillators that exhibits
# syncrhonization.  cf https://en.wikipedia.org/wiki/Kuramoto_model.
# We use this as an example to use jitcode. Latter, the same problem will be
# written in a somewhat more OOP approach, in a separate file
# "jitcode_oop_example.py".


# kuramoto model
NUM_NEURON = 10
NUM_DIM_NEURON = 1 # the dimension (no. of variables) of a single neuron.
NUM_SYN = 0 # For simplicity in the example we have no dynamical synapse.
NUM_DIM_SYN = 0 # the dimension (no. of variables) of a single synapse.
NUM_DIM_TOT = NUM_NEURON*NUM_DIM_NEURON + NUM_SYN*NUM_DIM_SYN
int_freqs = np.random.uniform(0.8,1.0,NUM_NEURON)
couple_const = 0.2 # Play with this number and see what happens!
# get adjacency matrix of some network
A = np.random.rand(NUM_NEURON,NUM_NEURON); A = A>0.5
# and equivalently an adjaceny list
A_list = [np.where(A[i,:])[0] for i in range(NUM_NEURON)]

# define ODEs
def f():
	for n in range(NUM_NEURON):
		coupling_sum = sum(symengine.sin(y(m)-y(n)) for m in A_list[n]) # nonlinearity!
		coupling_term = couple_const * coupling_sum
		yield  int_freqs[n] + coupling_term

# Impose initial conditions
initial_state = 2*np.random.random(NUM_DIM_TOT) - 1

# Magic begins
ODE = jitcode(f, n=NUM_DIM_TOT)
ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
ODE.set_integrator('dopri5')
ODE.set_initial_value(initial_state,0.0)

data = np.vstack(ODE.integrate(T) for T in np.arange(0., 10., 0.1))

# Time to witness magic
plt.figure()
for i in range(NUM_DIM_TOT):
    plt.plot(np.sin(data[:,i]))
plt.show()
