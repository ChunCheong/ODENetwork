from jitcode import jitcode, y, t
import symengine
import numpy as np
import matplotlib.pyplot as plt

def heaviside(x,K=1.e3):
    K = 1e3 # some big number
    return 1./(1.+ symengine.exp(-K*x))

def f():
    yield heaviside(t-1)

ODE = jitcode(f)
ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
ODE.set_integrator('dopri5')
ODE.set_initial_value([0.],)

ts = np.arange(0., 2., 0.1)
data = np.vstack(ODE.integrate(t) for t in ts)
plt.plot(ts, data)
plt.show()
