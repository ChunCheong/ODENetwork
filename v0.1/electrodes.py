"""
electrodes.py
A module that contains various kind of input currents.
To-dos:
1. make sym2num for mutiple arguments
"""
import numpy as np

from jitcode import t
try:
    import symengine as sym_backend
except:
    import sympy as sym_backend

def sigmoid(x):
    return 1./(1.+ sym_backend.exp(-x))

def heaviside(x):
    K = 1e3 # some big number
    return sigmoid(K*x)

def unit_pulse(t,t0,w):
    return heaviside(t-t0)*heaviside(w+t0-t)

"""
sym2num(expr):

Take an expression with symbolic object "t", like any time varying current
e.g. some_neuron.i_inj, to a normal function which is amenable to numerical
evaluation.

"""
def sym2num(t, expr):
    return sym_backend.Lambdify(t, expr)
