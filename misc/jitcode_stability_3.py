from jitcode import jitcode, y, t
import symengine
import numpy as np
import matplotlib.pyplot as plt

# observe what happens as you crank up K=1.e0 to K=1.e3
def heaviside(x,K=1.e0):
    return 1./(1.+ symengine.exp(-K*x))

def f():
    yield heaviside(t-1)

ODE = jitcode(f)
ODE.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150)
ODE.set_integrator('dopri5')
ODE.set_initial_value([0.],0.)

ts = np.arange(0., 2000., 1.) # testing underflow
data = np.vstack(ODE.integrate(t) for t in ts)
plt.plot(ts, data)
plt.show()
