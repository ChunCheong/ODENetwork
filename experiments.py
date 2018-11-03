"""
experiments.py

A collection of various experiments that you can do without killing an animal.
Feel free to put your experiment design here!

"""
import electrodes
from jitcode import t # symbolic time varibale, useful for defining currents


"""
a simple experiment playing with calcium-based STDP in a 2-layer FC network
"""
def delay_pulses_on_layer_1_and_2(net):
    i_max = 50. #5. # (some unit)
    t0 = 50. # ms
    dt = 10.
    w = 1. #ms
    for neuron in net.layer_1:
        neuron.i_inj = i_max*electrodes.unit_pulse(t,t0,w) # the jitcode t
    for neuron in net.layer_2:
        neuron.i_inj = i_max*electrodes.unit_pulse(t,t0+dt,w)
