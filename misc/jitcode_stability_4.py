from jitcode import jitcode, y, t
import symengine
import numpy as np
import matplotlib.pyplot as plt

def heaviside(x,K=1.e4): # K= some_big_number
    return 1./(1.+ symengine.exp(-K*x))

def unit_pulse(t,t0,w):
    return heaviside(t-t0)*heaviside(w+t0-t)

def f(): # a somewhat stiff problem
    t0, w, dt = 1., 0.00001,  0.0001
    yield 100000.*(unit_pulse(t,t0,w) - unit_pulse(t,t0+dt,w))

ODE = jitcode(f)
ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
ODE.set_integrator('dopri5')
ODE.set_initial_value([0.],0.)

ts = np.arange(0., 2000., 1.) # exp(K*2000) should really cause overflow
data = np.vstack(ODE.integrate(t) for t in ts)
plt.plot(ts, data)
plt.show()
