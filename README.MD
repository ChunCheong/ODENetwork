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


## Authors

Jonathan C.C. Lam [https://github.com/ChunCheong]
[https://www.linkedin.com/in/jonathan-chun-cheong-lam-707b25143]

(And put you name here if you want to join the band!)


## Acknowledgments

I would have something to say here if MSR is supporting me.
