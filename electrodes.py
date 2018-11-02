"""
electrodes.py
A module that contains various kind of input currents.
To-dos:
1. make sym2num for mutiple arguments
"""
from jitcode import jitcode, y, t
import numpy as np
import symengine

def sigmoid(x):
    return 1./(1.+ symengine.exp(-x))

def heaviside(x):
    K = 1e3 # some big number
    return sigmoid(K*x)

def unit_pulse(t,t0,w):
    return heaviside(t-t0)*heaviside(w+t0-t)

def sym2num(f):
    x = symengine.symbols('x')
    expr = f(x)
    return symengine.Lambdify(x, expr)
