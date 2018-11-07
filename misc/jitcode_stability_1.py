from jitcode import jitcode, y, t
import symengine
import numpy as np
import matplotlib.pyplot as plt

def heaviside(x,K=1.e3): # K= some_big_number
    return 1./(1.+ symengine.exp(-K*x))

def unit_pulse(t,t0,w):
    return heaviside(t-t0)*heaviside(w+t0-t)

def f():
    t0, w = 1., 0.00001
    yield 100000.*unit_pulse(t,t0,w)

ODE = jitcode(f)
ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
ODE.set_integrator('dopri5')
ODE.set_initial_value([0.],0.)

ts = np.arange(0., 2., 0.001)
data = np.vstack(ODE.integrate(t) for t in ts)
plt.plot(ts, data)
plt.show()
