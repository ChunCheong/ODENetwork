# ODENetwork

ODENetwork is a framework to simulate networks of ODEs, with emphasis to
model biological neural networks with dynamical neurons and synapses.
It was written with the intention to simply the workflow for studying  
Hebbian/reinforcement learning with a variety of neuron and synapse models.
While there are plenty of existing frameworks such as NEURON,
Brian and NEST, each with a slightly different focus. Being pythonic and feels
native is certainly one of the focuses of ODENetwork. Fortunately, speed is not
compromised thanks to the technology provided by JiTCODE. Finally, it is still
being actively developed. There would be inevitably bugs and typos so please
feel free to comment/contribute/test!


## Getting Started

Try running main.py to have a taste of what is it doing. If you think this is
your thing, then next you can define your own models.
You can put any kind of neuron models, network topologies and input protocol
under "neruon_models.py", "networks.py" and "experiments.py" respectively. In
most cases, you probably don't need to modify "lab_manager.py".
You might also want look at the examples under "/jitcode_example".


### Prerequisites

1. NetworkX (https://networkx.github.io/)

2. JiTCODE (https://jitcode.readthedocs.io/en/v1.3.0/#)

### Known Issues

There are known issues related to the behavior of integrators.
In particular, we observed issues when dealing with sharp changes such as
step/pules current. As suggested by the author of jitcode, we implement via a
sharp sigmoid. However, as observed in some combinations of various libraries,
the integrator is not able to handle that. Ultimately, jitcode uses scipy as
the backend and them compiles it into C code. It is not entirely clear to us
what to the proper way to treat that. Please also see various examples under
"/misc". Belows are some combinations of versions of python and relevant
libraries that is compatible with our code:

#### Linux:

##### Environment 1.
* Python: 3.6.4 |Anaconda, Inc.| (default, Mar 13 2018, 01:15:57) [GCC 7.2.0]
* Jitcode: 1.3.2
* scipy: 1.1.0
* networkx: 2.2

Any error message(s) from scipy, jitcode, integrators...etc? : None

##### Environment 2.
Python: 2.7.15 |Anaconda custom (64-bit)| (default, Oct 10 2018, 21:32:13) 
[GCC 7.3.0]
Jitcode: 0.61
scipy: 1.1.0
networkx: 2.2

anaconda2/lib/python2.7/site-packages/scipy/integrate/_ode.py:1177: UserWarning: dopri5: step size becomes too small

#### Mac

#### Windows:
##### Environment 1.
* Python: 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 10:22:32) [MSC v.1900 64 bit (AMD64)]
* Jitcode: 1.3.2
* scipy: 1.0.0
* networkx: 1.0.0

Any error message(s) from scipy, jitcode, integrators...etc? : None

Please kindly add your output from running "compat.py" above and tell us if you
see any error message when running. There are also some known issues with
matplotlib with Mac user which does not concern us here.


## Authors

Jonathan C.C. Lam [https://github.com/ChunCheong]
[https://www.linkedin.com/in/jonathan-chun-cheong-lam-707b25143]

David Li (UCSD)

(And put you name here if you want to join the band!)


## Acknowledgments

The author would like to thanks his two cats for warming his keyboard when
coding.
