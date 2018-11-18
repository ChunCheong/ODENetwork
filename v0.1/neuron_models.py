"""
neuron_models.py
A module that contains all the neuron and synapse model classes.
To-dos:
1. make method/function to check dimensions consistency across DIM, ic,...etc
2. introduce delay, related to choices for rho_gate placement
3. too many "self"... can improve readability?
"""
# begin boiler plate for compatibility
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
import sys
if sys.version_info.major > 2:
    xrange = range
elif sys.version_info.major == 2:
    pass
# end boiler plate for compatibility

from jitcode import jitcode, y, t
import numpy as np
try:
    import symengine as sym_backend
except:
    import sympy as sym_backend
# "Global" constants, if any

# Some very common helper functions
def sigmoid(x):
    return 1./(1.+ sym_backend.exp(-x))

def heaviside(x):
    K = 1e5 # some big number
    return sigmoid(K*x)

def pulse(t,t0,w):
    return heaviside(t-t0)*heaviside(w+t0-t)


class StaticSynapse:
    """
    A static synapse
    """
    # Dimension
    DIM = 0
    def __init__(self, syn_weight=1.):
        self.syn_weight = syn_weight

    # We should not need the followings for static object:
    # def fix_integration_index(self, i):
    #     pass
    # def dydt(self, pre_neuron, pos_neuron):
    #     pass
    # def get_initial_condition():
    #     pass
    #

class PlasticNMDASynapse:
    """
    A plastic synaspe
    """
    # Nernst/reversal potentials
    HF_PO_NMDA = 20 # NMDA half potential, unit: mV
    # Transmitter shit
    MAX_NMDA = 1. # it is always one!!! don't chnage it
    ALPHA_NMDA = 1.
    BETA_NMDA = 5.
    # Voltage response width (sigma)
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # time constants
    TAU_CA = 5.
    TAU_W = 1000.
    # stdp stuff
    THETA_P = 1.3
    THETA_D = 1.
    GAMMA_P = 220
    GAMMA_D = 100
    W_STAR = 0.5

    # Dimension
    DIM = 2
    def __init__(self,para=None):
        """
        Args:
            para: list of instance specific parameters
        """
        # self.rho_gate = y(i)
        # self.syn_weight = y(i+1)
        self.ii = None # integration index
        self.syn_weight = None #y(i)
        self.rho_gate = None

    def set_integration_index(self, i):
        self.ii = i # integration index
        self.syn_weight = y(i)
        self.rho_gate = y(i+1)

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        ca_pos = pos_neuron.calcium
        rho = self.rho_gate # some choice has to be made here
        #rho = pre_neuron.rho_gate
        wij = self.syn_weight
        t_conc = self.MAX_NMDA*sigmoid((v_pre-self.HF_PO_NMDA)/self.V_REW_NMDA)
        yield 1./self.TAU_W*(
            - wij*(1-wij)*(self.W_STAR-wij)
            + self.GAMMA_P*(1-wij)*heaviside(ca_pos - self.THETA_P)
            - self.GAMMA_D*wij*heaviside(ca_pos - self.THETA_D) )
        yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho

    def get_initial_condition(self):
        return [np.random.rand(), 0.]

class PlasticNMDASynapseWithCa:
    """
    A plastic synaspe
    """
    # Nernst/reversal potentials
    HF_PO_NMDA = 20 # NMDA half potential, unit: mV
    # Transmitter shit
    MAX_NMDA = 1. # it is always one!!! don't chnage it
    ALPHA_NMDA = 1.
    BETA_NMDA = 5.
    # Voltage response width (sigma)
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # CALCIUM
    CA_EQM = 0.
    AVO_CONST = 0.03 # "Avogadros" constant, relate calcium concentraion and current
    # time constants
    TAU_CA = 5.
    TAU_W = 1000.
    # stdp stuff
    THETA_P = 1.3
    THETA_D = 1.
    GAMMA_P = 220
    GAMMA_D = 100
    W_STAR = 0.5

    # Dimension
    #DIM = 2
    DIM = 3
    def __init__(self,para=None):
        """
        Args:
            para: list of instance specific parameters
        """
        # self.rho_gate = y(i)
        # self.syn_weight = y(i+1)
        self.ii = None # integration index
        self.syn_weight = None #y(i)
        self.rho_gate = None
        self.ca = None

    def set_integration_index(self, i):
        self.ii = i # integration index
        self.syn_weight = y(i)
        self.rho_gate = y(i+1)
        self.ca = y(i+2) ###

    def dydt(self, pre_neuron, pos_neuron):
        v_pre = pre_neuron.v_mem
        v_pos = pos_neuron.v_mem ###
        a_pos = pos_neuron.a_gate ###
        b_pos = pos_neuron.b_gate ###
        i_syn_ca = pos_neuron.i_syn_ca_ij(v_pos, self.rho_gate, self.syn_weight) ###
        #ca_pos = pos_neuron.calcium
        rho = self.rho_gate # some choice has to be made here
        #rho = pre_neuron.rho_gate
        wij = self.syn_weight
        t_conc = self.MAX_NMDA*sigmoid((v_pre-self.HF_PO_NMDA)/self.V_REW_NMDA)
        yield 1./self.TAU_W*(
            - wij*(1-wij)*(self.W_STAR-wij)
            + self.GAMMA_P*(1-wij)*heaviside(self.ca - self.THETA_P)
            - self.GAMMA_D*wij*heaviside(self.ca - self.THETA_D) )
        yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho
        yield self.AVO_CONST*(pos_neuron.i_ca(v_pos, a_pos, b_pos) + i_syn_ca) + (self.CA_EQM-self.ca)/self.TAU_CA ###

    def get_initial_condition(self):
        return [0.5+ 0.1*np.random.rand(), 0., 0.] ###


class HHNeuronWithCa:
    """
    Actually a slight variations of the canonical Hodgkin-Huxley neuron.
    Contains a small calcium ion channel.
    Also the original motivation was to treat all gating variables on equal
    footing so that their taus and x_eqm have the same functional form. It
    probably does not matter much...?
    """
    # Parameters:
    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2
    # Conductances
    COND_LEAK = 0.3 # Max. leak conductance, unit: mScm^-2
    COND_NA = 120 # Max. Na conductance, unit: mScm^-2
    COND_K = 36 # Max. K conductance, unit: mScm^-2
    COND_CA = 1. # Max. Ca conductance, unit: mScm^-2
    COND_SYN = 3. #?
    COND_CA_SYN = 1.5
    # Nernst/reversal potentials
    RE_PO_LEAK = -70 # Leak Nernst potential, unit: mV
    RE_PO_NA = 50 # Na Nernst potential, unit: mV
    RE_PO_K = -95 # K Nernst potential, unit: mV
    RE_PO_CA = 140 # K Nernst potential, unit: mV
    RE_PO_SYN = 0.
    # Half potentials of gating variables
    HF_PO_M = -40 # m half potential, unit: mV
    HF_PO_H = -60 # h half potential, unit: mV
    HF_PO_N = -55 # n half potential, unit: mV
    HF_PO_A = -20#-70 # a half potential, unit: mV
    HF_PO_B = -25 #-65 # b half potential, unit: mV
    # Transmitter shit
    MAX_NMDA = 1.
    ALPHA_NMDA = 1.
    BETA_NMDA = 5.
    # Voltage response width (sigma)
    V_REW_M = 16 # m voltage response width, unit: mV
    V_REW_H = -16 # m voltage response width, unit: mV
    V_REW_N = 25 # m voltage response width, unit: mV
    V_REW_A = 13 #10 # m voltage response width, unit: mV
    V_REW_B = -24#-10 # m voltage response width, unit: mV
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # time constants
    TAU_0_M = 0.1 # unit ms
    TAU_1_M = 0.4
    TAU_0_H = 1.
    TAU_1_H = 7.
    TAU_0_N = 1.
    TAU_1_N = 5.
    TAU_0_A = 0.1
    TAU_1_A = 0.2
    TAU_0_B = 1.
    TAU_1_B = 5.
    TAU_CA = 5.
    # CALCIUM
    CA_EQM = 0.
    AVO_CONST = 0.014085831147459489 # DONT CHANGE IT # "Avogadros" constant, relate calcium concentraion and current

    # Dimension
    DIM = 7 #8 if exclude rho gate
    def __init__(self,para=None):
        """
        Put all the internal variables and instance specific constants here
        Examples of varibales include Vm, gating variables, calcium ...etc
        Constants can be various conductances, which can vary across
        instances.
        Args:
            para: list of instance specific parameters
            i_inj: injected current
        """
        self.i_inj = 0 # injected currents
        self.ii = None # integration index
        self.ni = None # neruon index
        self.v_mem = None #y(i) # membrane potential
        self.m_gate = None #y(i+1)
        self.n_gate = None #y(i+2)
        self.h_gate = None #y(i+3)
        self.a_gate = None #y(i+4)
        self.b_gate = None #y(i+5)
        self.calcium = None #y(i+6)
        #self.rho_gate = None #y(i+7)
        # may put para here

    def set_integration_index(self, i):
        """
        Args:
            i: integration variable index
        """
        self.ii = i # integration index
        self.v_mem = y(i) # membrane potential
        self.m_gate = y(i+1)
        self.n_gate = y(i+2)
        self.h_gate = y(i+3)
        self.a_gate = y(i+4)
        self.b_gate = y(i+5)
        self.calcium = y(i+6)
        #self.rho_gate = y(i+7)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        # define how neurons are coupled here
        v = self.v_mem
        m = self.m_gate
        n = self.n_gate
        h = self.h_gate
        a = self.a_gate
        b = self.b_gate
        ca = self.calcium
        #rho = self.rho_gate
        i_inj = self.i_inj
        i_leak = self.i_leak
        i_na = self.i_na
        # i_syn = sum(
        #     self.i_syn_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        # i_syn_ca = sum(
        #     self.i_syn_ca_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        i_syn = sum(
            self.i_syn_ij(v, synapse.rho_gate, synapse.syn_weight)
            for (i,synapse) in enumerate(pre_synapses) )
        i_syn_ca = sum(
            self.i_syn_ca_ij(v, synapse.rho_gate, synapse.syn_weight)
            for (i,synapse) in enumerate(pre_synapses) )
        i_base = (
            i_syn + self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n)
            + self.i_ca(v,a,b) )
        if i_inj is None:
            yield 1/self.CAP_MEM*i_base
        else:
            yield 1/self.CAP_MEM*(i_inj+i_base)
        yield 1/self.tau_x(
            v, self.HF_PO_M, self.V_REW_M, self.TAU_0_M, self.TAU_1_M
            )*(self.x_eqm(v, self.HF_PO_M, self.V_REW_M) - m)
        yield 1/self.tau_x(
            v, self.HF_PO_N, self.V_REW_N, self.TAU_0_N, self.TAU_1_N
            )*(self.x_eqm(v, self.HF_PO_N, self.V_REW_N) - n)
        yield 1/self.tau_x(
            v, self.HF_PO_H, self.V_REW_H, self.TAU_0_H, self.TAU_1_H
            )*(self.x_eqm(v, self.HF_PO_H, self.V_REW_H) - h)
        yield 1/self.tau_x(
            v, self.HF_PO_A, self.V_REW_A, self.TAU_0_A, self.TAU_1_A
            )*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        yield 1/self.tau_x(
            v, self.HF_PO_B, self.V_REW_B, self.TAU_0_B, self.TAU_1_B
            )*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        yield self.AVO_CONST*(
            self.i_ca(v,m,h) + i_syn_ca) + (self.CA_EQM-ca)/self.TAU_CA
        #yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho

    def get_initial_condition(self):
        return [-73.,0.2,0.8,0.2,0.2,0.8,0.]

    # some helper functions for dydt
    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    # def t_eqm(self, T):
    #     return ALPHA_NMDA*T/(ALPHA_NMDA*T + BETA_NMDA)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    # def tau_syn(T):
    #     return 1./(ALPHA_NMDA*T + BETA_NMDA)
    #@staticmethod
    def i_leak(self, Vm):
        return -self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return -self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return -self.COND_K*n**4*(Vm - self.RE_PO_K)

    def i_ca(self, Vm, a, b):
        return -self.COND_CA*a**2*b*(Vm - self.RE_PO_CA)

    def i_syn_ij(self, Vm_po, rho_ij, W_ij):
        return - self.COND_SYN*W_ij*rho_ij*(Vm_po - self.RE_PO_SYN)

    def i_syn_ca_ij(self, Vm_po, rho_ij, W_ij):
        return - self.COND_CA_SYN*0.5*rho_ij*(Vm_po - self.RE_PO_CA)

class PlasticNMDASynapseWithCaJL:
    """
    A plastic synaspe inspired by Graypner and Brunel (2012).
    The model used by them has a limited dynamical range of synaptic weight
    fixed the ratio GAMMA_P/(GAMMA_D + GAMMA_P). We relaxed that by a
    modification to the eom of synaptic weight.
    """
    # Nernst/reversal potentials
    HF_PO_NMDA = 20 # NMDA half potential, unit: mV
    RE_PO_CA = 140 # K Nernst potential, unit: mV treating the same as neuron
    # Transmitter shit
    ALPHA_NMDA = 10. # just make it as big as possible so that rho_max is one
    # Voltage response width (sigma)
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # CALCIUM
    CA_EQM = 0.
    RELATIVE_COND_CA_SYN = 1.
    # This choices normalizes both synapse- and voltage-gated calcium peak to 1.
    AVO_CONST_SYN = 0.00095#0.0002 # ~"Avogadros constant", relate calcium concentraion and current
    AVO_CONST_POS = 0.013972995788339456
    #COND_CA_SYN = RELATIVE_COND_CA_SYN*5.119453924914676#1.5
    # time constants
    TAU_CA = 5.
    #TAU_W = 10000.
    TAU_RHO = 1.5*TAU_CA
    # stdp stuff
    # THETA_* are measured in unit of voltage-gated calcium peak = 1
    THETA_P = 0.85
    THETA_D = 0.4
    GAMMA = 0.1
    # if has zero rise time
    # GAMMA_D = 1*np.log(THETA_P)/np.log(THETA_D) # have to be calibtated
    # finite rise time: need calibration
    GAMMA_D = 0.23389830508474577
    # Dimension
    #DIM = 2
    DIM = 3
    def __init__(self,para=None):
        """
        Args:
            para: list of instance specific parameters
        """
        # self.rho_gate = y(i)
        # self.syn_weight = y(i+1)
        self.ii = None # integration index
        self.reduced_weight = None
        self.rho_gate = None
        self.ca = None

    def set_integration_index(self, i):
        self.ii = i # integration index
        #self.syn_weight = y(i)
        self.reduced_weight = y(i)
        self.rho_gate = y(i+1)
        self.ca = y(i+2) ###

    def dydt(self, pre_neuron, pos_neuron):
        # gating varibales
        v_pre = pre_neuron.v_mem
        v_pos = pos_neuron.v_mem ###
        # a_pos = pos_neuron.a_gate ###
        # b_pos = pos_neuron.b_gate ###
        rw = self.reduced_weight
        #wij = self.syn_weight()
        rho = self.rho_gate
        ca = self.ca
        # calcium currents
        i_syn_ca = self.AVO_CONST_SYN*self.i_syn_ca_ij(pos_neuron)
        i_pos_ca = self.AVO_CONST_POS*pos_neuron.i_ca()
        i_leak_ca = (self.CA_EQM-self.ca)/self.TAU_CA
        # transmitter (only NMDA here)
        t_conc = sigmoid((v_pre-self.HF_PO_NMDA)/self.V_REW_NMDA)
        # derivatives
        yield self.GAMMA*(
            heaviside(ca- self.THETA_P)
            -self.GAMMA_D*heaviside(ca - self.THETA_D))
        yield self.ALPHA_NMDA*t_conc*(1-rho) - rho/self.TAU_RHO
        yield i_syn_ca + i_pos_ca + i_leak_ca
        # yield self.AVO_CONST*( pos_neuron.i_ca(-70., a_pos, b_pos)
        #     + i_syn_ca) + (self.CA_EQM-self.ca)/self.TAU_CA ###
    # helper functions
    # The synaptic current at this particular dendrite/synapse
    # It should depends only on the pos-synaptic voltage
    def i_syn_ca_ij(self, pos_neuron):
        v_pos = pos_neuron.v_mem ###
        rho = self.rho_gate
        wij = self.syn_weight()
        #return - self.COND_CA_SYN*rho_ij*(Vm_po - self.RE_PO_CA)
        return - wij*rho*(v_pos - pos_neuron.RE_PO_CA)

    def i_syn_ij(self, pos_neuron):
        v_pos = pos_neuron.v_mem ###
        rho = self.rho_gate
        wij = self.syn_weight()
        return - wij*rho*(v_pos - pos_neuron.RE_PO_SYN)

    def syn_weight(self):
        return sigmoid(self.reduced_weight)

    def get_initial_condition(self):
        return [0., 0., 0.] ###


class HHNeuronWithCaJL:
    """
    Actually a slight variations of the canonical Hodgkin-Huxley neuron.
    Originally we added calcium as a perturbation, which is not important
    in the neuron dynamics anyway. Here the ca is just tagging along, so are
    the relevant gating varibales. They are important in the synaptic activity
    but not neuron activity.
    Also the original motivation was to treat all gating variables on equal
    footing so that their taus and x_eqm have the same functional form. It
    probably does not matter much...?
    """
    # Parameters:
    # Capacitance
    CAP_MEM = 1. # membrane capacitance, unit: uFcm^-2
    # Conductances
    COND_LEAK = 0.3 # Max. leak conductance, unit: mScm^-2
    COND_NA = 120 # Max. Na conductance, unit: mScm^-2
    COND_K = 36 # Max. K conductance, unit: mScm^-2
    COND_CA = 1. # Max. Ca conductance, unit: mScm^-2
    COND_SYN = .5 # have to be fiine tuned according to each network
    #COND_CA_SYN = 1.5
    # Nernst/reversal potentials
    RE_PO_LEAK = -70 # Leak Nernst potential, unit: mV
    RE_PO_NA = 50 # Na Nernst potential, unit: mV
    RE_PO_K = -95 # K Nernst potential, unit: mV
    RE_PO_CA = 140 # K Nernst potential, unit: mV
    RE_PO_SYN = 0.
    # Half potentials of gating variables
    HF_PO_M = -40 # m half potential, unit: mV
    HF_PO_H = -60 # h half potential, unit: mV
    HF_PO_N = -55 # n half potential, unit: mV
    HF_PO_A = -20#-70 # a half potential, unit: mV
    HF_PO_B = -25 #-65 # b half potential, unit: mV
    # Voltage response width (sigma)
    V_REW_M = 16 # m voltage response width, unit: mV
    V_REW_H = -16 # m voltage response width, unit: mV
    V_REW_N = 25 # m voltage response width, unit: mV
    V_REW_A = 13 #10 # m voltage response width, unit: mV
    V_REW_B = -24#-10 # m voltage response width, unit: mV
    V_REW_NMDA = 2 # NMDA voltage response width, unit: mV
    # time constants
    TAU_0_M = 0.1 # unit ms
    TAU_1_M = 0.4
    TAU_0_H = 1.
    TAU_1_H = 7.
    TAU_0_N = 1.
    TAU_1_N = 5.
    TAU_0_A = 0.1
    TAU_1_A = 0.2
    TAU_0_B = 1.
    TAU_1_B = 5.
    TAU_CA = 5.
    # CALCIUM
    #CA_EQM = 0.
    #AVO_CONST = 0.014085831147459489 # DONT CHANGE IT # "Avogadros" constant, relate calcium concentraion and current

    # Dimension
    DIM = 6 #8 if exclude rho gate
    def __init__(self,para=None):
        """
        Put all the internal variables and instance specific constants here
        Examples of varibales include Vm, gating variables, calcium ...etc
        Constants can be various conductances, which can vary across
        instances.
        Args:
            para: list of instance specific parameters
            i_inj: injected current
        """
        self.i_inj = 0 # injected currents
        self.ii = None # integration index
        self.ni = None # neruon index
        self.v_mem = None #y(i) # membrane potential
        self.m_gate = None #y(i+1)
        self.n_gate = None #y(i+2)
        self.h_gate = None #y(i+3)
        self.a_gate = None #y(i+4)
        self.b_gate = None #y(i+5)
        #self.calcium = None #y(i+6)
        #self.rho_gate = None #y(i+7)
        # may put para here

    def set_integration_index(self, i):
        """
        Args:
            i: integration variable index
        """
        self.ii = i # integration index
        self.v_mem = y(i) # membrane potential
        self.m_gate = y(i+1)
        self.n_gate = y(i+2)
        self.h_gate = y(i+3)
        self.a_gate = y(i+4)
        self.b_gate = y(i+5)
        #self.calcium = y(i+6)
        #self.rho_gate = y(i+7)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons):
        # define how neurons are coupled here
        v = self.v_mem
        m = self.m_gate
        n = self.n_gate
        h = self.h_gate
        a = self.a_gate
        b = self.b_gate
        #ca = self.calcium
        #rho = self.rho_gate
        i_inj = self.i_inj
        i_leak = self.i_leak
        i_na = self.i_na
        # i_syn = sum(
        #     self.i_syn_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        # i_syn_ca = sum(
        #     self.i_syn_ca_ij(v, pre_neurons[i].rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        i_syn = self.COND_SYN*sum(synapse.i_syn_ij(self)
            for (i,synapse) in enumerate(pre_synapses))
        # i_syn_ca = sum(
        #     self.i_syn_ca_ij(v, synapse.rho_gate, synapse.syn_weight)
        #     for (i,synapse) in enumerate(pre_synapses) )
        i_base = i_syn + self.i_leak(v) + self.i_na(v,m,h) + self.i_k(v,n)
            #+ self.i_ca(v,a,b) )
        if i_inj is None:
            yield 1/self.CAP_MEM*i_base
        else:
            yield 1/self.CAP_MEM*(i_inj+i_base)
        yield 1/self.tau_x(
            v, self.HF_PO_M, self.V_REW_M, self.TAU_0_M, self.TAU_1_M
            )*(self.x_eqm(v, self.HF_PO_M, self.V_REW_M) - m)
        yield 1/self.tau_x(
            v, self.HF_PO_N, self.V_REW_N, self.TAU_0_N, self.TAU_1_N
            )*(self.x_eqm(v, self.HF_PO_N, self.V_REW_N) - n)
        yield 1/self.tau_x(
            v, self.HF_PO_H, self.V_REW_H, self.TAU_0_H, self.TAU_1_H
            )*(self.x_eqm(v, self.HF_PO_H, self.V_REW_H) - h)
        yield 1/self.tau_x(
            v, self.HF_PO_A, self.V_REW_A, self.TAU_0_A, self.TAU_1_A
            )*(self.x_eqm(v, self.HF_PO_A, self.V_REW_A) - a)
        yield 1/self.tau_x(
            v, self.HF_PO_B, self.V_REW_B, self.TAU_0_B, self.TAU_1_B
            )*(self.x_eqm(v, self.HF_PO_B, self.V_REW_B) - b)
        # yield self.AVO_CONST*(
        #     self.i_ca(v,m,h) + i_syn_ca) + (self.CA_EQM-ca)/self.TAU_CA
        #yield self.ALPHA_NMDA*t_conc*(1-rho) - self.BETA_NMDA*rho

    def get_initial_condition(self):
        return [-73.,0.2,0.8,0.2,0.2,0.8]

    # some helper functions for dydt
    def x_eqm(self, Vm, V_0, sigma_x):
        return sigmoid(2*(Vm - V_0)/sigma_x)

    # def t_eqm(self, T):
    #     return ALPHA_NMDA*T/(ALPHA_NMDA*T + BETA_NMDA)

    def tau_x(self, Vm, V_0, sigma_x, tau_x_0, tau_x_1):
        return tau_x_0 + tau_x_1*(1-(sym_backend.tanh((Vm - V_0)/sigma_x))**2)

    # def tau_syn(T):
    #     return 1./(ALPHA_NMDA*T + BETA_NMDA)
    #@staticmethod
    def i_leak(self, Vm):
        return -self.COND_LEAK*(Vm - self.RE_PO_LEAK)

    def i_na(self, Vm, m, h):
        return -self.COND_NA*m**3*h*(Vm - self.RE_PO_NA)

    def i_k(self, Vm, n):
        return -self.COND_K*n**4*(Vm - self.RE_PO_K)

    # def i_ca(self, Vm, a, b):
    #     return -self.COND_CA*a**2*b*(Vm - self.RE_PO_CA)
    def i_ca(self):
        v = self.v_mem
        a = self.a_gate
        b = self.b_gate
        return -self.COND_CA*a**2*b*(v-self.RE_PO_CA)

    # def i_syn_ij(self, Vm_po, rho_ij, W_ij):
    #     return - self.COND_SYN*W_ij*rho_ij*(Vm_po - self.RE_PO_SYN)

    # def i_syn_ca_ij(self, Vm_po, rho_ij, W_ij):
    #     return - self.COND_CA_SYN*0.5*rho_ij*(Vm_po - self.RE_PO_CA)

"Hodgkin-Huxley Neuron with paramaters from Henry\
Defined uA/cm^2"
class HHNeuron:
    # Constants
    C_m  =   1.0 # membrane capacitance, in uF/cm^2

    # maximum conducances, in mS/cm^2
    g_Na =   120.0
    g_K  =   20.0
    g_L  =   0.3

    # Nernst reversal potentials, in mV
    E_Na = 50.0
    E_K  = -77.0
    E_L  = -54.4

    # kinetics, mv
    vm = -40.0
    vn = -55.0
    vh = -60.0

    dvm = 15.0
    dvn = 30.0
    dvh = -15.0

    vmt = -40.0
    vnt = -55.0
    vht = -60.0

    dvmt = 15.0
    dvnt = 30.0
    dvht = -15.0

    #ms
    tm0 = 0.1
    tm1 = 0.4

    tn0 = 1.0
    tn1 = 5.0

    th0 = 1.0
    th1 = 7.0

    DIM = 4

    def __init__(self, para = None):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        self.i_inj = 0.0 # injected currents
        self.V = None
        self.m = None
        self.h = None
        self.n = None

    #H-H model
    def set_integration_index(self, i):
        self.ii = i
        self.V = y(i)
        self.m = y(i+1)
        self.h = y(i+2)
        self.n = y(i+3)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons): # a list of pre-synaptic neurons
        # define how neurons are coupled here
        VV = self.V
        mm = self.m
        hh = self.h
        nn = self.n
        i_inj = self.i_inj
        i_syn = sum(self.I_syn(VV, y(synapse.get_ind()),
                    synapse.get_params()[0], synapse.get_params()[1], synapse.weight)
                    for (i,synapse) in enumerate(pre_synapses))

        i_base = (self.I_Na(VV, mm, hh) + self.I_K(VV, nn) +
                            self.I_L(VV) + i_syn)

        yield -1/self.C_m*(i_inj+i_base)
        yield self.dm_dt(VV, mm)
        yield self.dh_dt(VV, hh)
        yield self.dn_dt(VV, nn)


    def get_initial_condition(self):
        return [-65.0, 0.05, 0.6, 0.32]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.V

    def I_syn(self, V, r, gNt, E_nt, w): return gNt*r*w*(V - E_nt)

    def m0(self, V): return 0.5*(1+sym_backend.tanh((V - self.vm)/self.dvm))
    def n0(self, V): return 0.5*(1+sym_backend.tanh((V - self.vn)/self.dvn))
    def h0(self, V): return 0.5*(1+sym_backend.tanh((V - self.vh)/self.dvh))

    def tau_m(self, V): return self.tm0+self.tm1*(1-sym_backend.tanh((V - self.vmt)/self.dvmt)**2)
    def tau_n(self, V): return self.tn0+self.tn1*(1-sym_backend.tanh((V - self.vnt)/self.dvnt)**2)
    def tau_h(self, V): return self.th0+self.th1*(1-sym_backend.tanh((V - self.vht)/self.dvht)**2)

    def I_Na(self, V, m, h): return self.g_Na*m**3*h*(V - self.E_Na) #mS*mV = uA
    def I_K(self, V, n): return self.g_K*n**4*(V - self.E_K)
    def I_L(self, V): return self.g_L*(V - self.E_L)

    def dV_dt(self, V, m, h, n, t, i_syn):
        return -1/self.C_m *(self.I_Na(V, m, h) + self.I_K(V, n) +
                            self.I_L(V) - self.i_inj + i_syn)

    def dm_dt(self, V, m): return (self.m0(V) - m)/self.tau_m(V)
    def dh_dt(self, V, h): return (self.h0(V) - h)/self.tau_h(V)
    def dn_dt(self, V, n): return (self.n0(V) - n)/self.tau_n(V)


class Synapse_glu_HH:
    #Excitation
    Gglu = 0.4
    E_cl = -38.0 #mV
    alphaR = 2.4
    betaR = 0.56
    Tm = 1.0


    Kp = 5.0
    Vp = 7.0

    DIM = 1
    def __init__(self, para = None):
        self.r = None
        self.weight = 1.0
    def set_integration_index(self, i):
        self.ii = i
        self.r = y(i)

    def fix_weight(self, w):
        self.weight = w

    def dydt(self, pre_neuron, pos_neuron):
        Vpre = pre_neuron.V
        r = self.r
        yield (self.alphaR*self.Tm/(1+sym_backend.exp(-(Vpre - self.Vp)/self.Kp)))*(1-r) - self.betaR*r
    def get_params(self):
        return [self.Gglu, self.E_cl]

    def get_ind(self):
        return self.ii
    def get_initial_condition(self):
        return [0.1]

class Synapse_gaba_HH:
    #inhibition
    gGABA = 1.0
    E_gaba = -80.0 #mV
    alphaR = 5.0
    betaR = 0.18
    Tm = 1.5


    Kp = 5.0
    Vp = 7.0

    DIM = 1
    def __init__(self, para = None):
        self.r = None
        self.weight = 1.0

    def set_integration_index(self, i):
        self.ii = i
        self.r = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.weight = w

    def dydt(self, pre_neuron, pos_neuron):
        Vpre = pre_neuron.V
        r = self.r
        yield (self.alphaR*self.Tm/(1+sym_backend.exp(-(Vpre - self.Vp)/self.Kp)))*(1-r) - self.betaR*r

    def get_params(self):
        return [self.gGABA, self.E_gaba]

    def get_initial_condition(self):
        return [0.1]

"Fitted Model of Projection Neurons from the Bazhenov Papers\
Defined in pico amps"
class PN_2:
    # Constants for PNs
    C_m  =   142.0 # membrane capacitance, in pF

    # maximum conducances, in nS
    g_Na_PN =   7150.0
    g_K_PN  =   1430.0
    g_L_PN  =   21.0
    g_KL_PN =   5.72
    g_A_PN  =   1430.0

    # Nernst reversal potentials, in mV
    E_Na_PN = 50.0
    E_K_PN  = -95.0
    E_L_PN  = -55.0
    E_KL_PN = -95.0


    # Gating Variable m parameters
    HF_PO_M = -43.9
    V_REW_M = -7.4
    HF_PO_MT = -47.5
    V_REW_MT = 40.0
    TAU_0_M = 0.024
    TAU_1_M = 0.093

    # Gating Variable h Parameters
    HF_PO_H = -48.3
    V_REW_H = 4.0
    HF_PO_HT = -56.8
    V_REW_HT = 16.9
    TAU_0_H = 0.0
    TAU_1_H = 5.6

    shift = 70.0

    DIM = 6

    def __init__(self, para = None):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        self.i_inj = 0.0 # injected currents
        self.V = None
        self.m = None
        self.h = None
        self.n = None
        self.z = None
        self.u = None

    def set_integration_index(self, i):
        self.ii = i
        self.V = y(i)
        self.m = y(i+1)
        self.h = y(i+2)
        self.n = y(i+3)
        self.z = y(i+4)
        self.u = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons): # a list of pre-synaptic neurons
        # define how neurons are coupled here
        VV = self.V
        mm = self.m
        hh = self.h
        nn = self.n
        zz = self.z
        uu = self.u
        i_inj = self.i_inj



        i_syn = sum(self.I_syn(VV, y(synapse.get_ind()),
                    synapse.get_params()[0], synapse.get_params()[1], synapse.weight)
                    for (i,synapse) in enumerate(pre_synapses))
        i_base = (self.I_Na(VV, mm, hh) + self.I_K(VV, nn) +
                        self.I_L(VV) + self.I_A(VV,zz,uu) + self.I_KL(VV)
                        + i_syn)

        yield -1/self.C_m*(i_base-i_inj)
        yield self.dm_dt(VV, mm)
        yield self.dh_dt(VV, hh)
        yield self.dn_dt(VV, nn)
        yield self.dz_dt(VV, zz)
        yield self.du_dt(VV, uu)


    def get_initial_condition(self):
        return [-65.0, 0.05, 0.6, 0.32, 0.6, 0.6]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.V

    def I_syn(self, V, r, gNt, E_nt, w): return gNt*r*w*(V - E_nt)

    def x_eqm(self,V,theta,sigma): return 0.5*(1.0 - sym_backend.tanh(0.5*(V-theta)/sigma))
    def tau_x(self,V,theta,sigma,t0,t1): return t0 + t1*(1.0-sym_backend.tanh((V-theta)/sigma)**2)

    def dm_dt(self,V,m): return (self.m0(V)-m)/self.tm(V)
    def dh_dt(self,V,h): return (self.h0(V)-h)/self.th(V)
    def dn_dt(self, V, n): return self.a_n(V)*(1-n)-self.b_n(V)*n
    def dz_dt(self, V, z): return (self.z0(V)-z)/self.tz(V)
    def du_dt(self, V, u): return (self.u0(V)-u)/self.tu(V)

    def m0(self,V): return self.x_eqm(V,self.HF_PO_M,self.V_REW_M)
    def tm(self,V): return self.tau_x(V,self.HF_PO_MT,self.V_REW_MT,self.TAU_0_M,self.TAU_1_M)

    def h0(self,V): return self.x_eqm(V,self.HF_PO_H,self.V_REW_H)
    def th(self,V): return self.tau_x(V,self.HF_PO_HT,self.V_REW_HT,self.TAU_0_H,self.TAU_1_H)

    def a_n(self, V): return 0.016*(V-35.1+self.shift)/(1-sym_backend.exp(-(V-35.1+self.shift)/5.0))
    def b_n(self, V): return 0.25*sym_backend.exp(-(V-20+self.shift)/40.0)

    def z0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+60)/8.5))
    def tz(self, V): return 0.27/sym_backend.exp((V+35.8)/19.7)+sym_backend.exp(-(V+79.7)/12.7)+0.1

    def u0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+78)/6.0))

    #adapted from bazhenov
    def tu(self, V):
        return 0.27/(sym_backend.exp((V+46)/5.0)+sym_backend.exp(-(V+238)/37.5)) \
                    +5.1/2*(1+sym_backend.tanh((V+57)/3))

    def I_Na(self, V, m, h): return self.g_Na_PN*m**3*h*(V - self.E_Na_PN) #nS*mV = pA
    def I_K(self, V, n): return self.g_K_PN*n*(V - self.E_K_PN)
    def I_L(self, V): return self.g_L_PN*(V - self.E_L_PN)
    def I_A(self, V, z, u): return self.g_A_PN*z**4*u*(V - self.E_K_PN)
    def I_KL(self, V): return self.g_KL_PN*(V - self.E_KL_PN)


class PN:
        # Constants for PNs
    C_m  =   142.0 # membrane capacitance, in pF

    # maximum conducances, in nS
    g_Na_PN =   7150.0
    g_K_PN  =   1430.0
    g_L_PN  =   21.0
    g_KL_PN =   5.72
    g_A_PN  =   1430.0

    # Nernst reversal potentials, in mV
    E_Na_PN = 50.0
    E_K_PN  = -95.0
    E_L_PN  = -55.0
    E_KL_PN = -95.0

    shift = 70.0
    DIM = 6

    def __init__(self, para = None):
        # Put all the internal variables and instance specific constants here
        # Examples of varibales include Vm, gating variables, calcium ...etc
        # Constants can be variouse conductances, which can vary across
        # instances.
        self.i_inj = 0.0 # injected currents
        self.V = None
        self.m = None
        self.h = None
        self.n = None
        self.z = None
        self.u = None

    def set_integration_index(self, i):
        self.ii = i
        self.V = y(i)
        self.m = y(i+1)
        self.h = y(i+2)
        self.n = y(i+3)
        self.z = y(i+4)
        self.u = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons): # a list of pre-synaptic neurons
    # define how neurons are coupled here
        VV = self.V
        mm = self.m
        hh = self.h
        nn = self.n
        zz = self.z
        uu = self.u
        i_inj = self.i_inj

        i_syn = sum(self.I_syn(VV, y(synapse.get_ind()),
                    synapse.get_params()[0], synapse.get_params()[1], synapse.weight)
                    for (i,synapse) in enumerate(pre_synapses))
        i_base = (self.I_Na(VV, mm, hh) + self.I_K(VV, nn) +
                    self.I_L(VV) + self.I_A(VV,zz,uu) + self.I_KL(VV)
                    + i_syn)

        yield -1/self.C_m*(i_base-i_inj)
        yield self.dm_dt(VV, mm)
        yield self.dh_dt(VV, hh)
        yield self.dn_dt(VV, nn)
        yield self.dz_dt(VV, zz)
        yield self.du_dt(VV, uu)


    def get_initial_condition(self):
        return [-65.0, 0.05, 0.6, 0.32, 0.6, 0.6]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.V

    def I_syn(self, V, r, gNt, E_nt, w): return gNt*r*w*(V - E_nt)

    def dV_dt(self, V, m, h, n, z, u, t, i_syn):
        return -1/self.C_m *(self.I_Na(V, m, h) + self.I_K(V, n) +
                        self.I_L(V) + self.I_A(V,z,u) + self.I_KL(V)
                         - self.i_inj + i_syn)
    def dm_dt(self, V, m): return self.a_m(V)*(1-m)-self.b_m(V)*m
    def dh_dt(self, V, h): return self.a_h(V)*(1-h)-self.b_h(V)*h
    def dn_dt(self, V, n): return self.a_n(V)*(1-n)-self.b_n(V)*n
    def dz_dt(self, V, z): return (self.z0(V)-z)/self.tz(V)
    def du_dt(self, V, u): return (self.u0(V)-u)/self.tu(V)

    def a_m(self, V): return 0.32*(V - 13.1+self.shift)/(1 - sym_backend.exp(-(V - 13.1+self.shift)/4.0))
    def b_m(self, V): return 0.28*(V - 40.1+self.shift)/(sym_backend.exp((V-40.1+self.shift)/5.0)-1)

    def a_h(self, V): return 0.128*sym_backend.exp(-(V-17.0+self.shift)/18.0)
    def b_h(self, V): return 4.0/(1+sym_backend.exp(-(V-40+self.shift)/5.0))

    def a_n(self, V): return 0.016*(V-35.1+self.shift)/(1-sym_backend.exp(-(V-35.1+self.shift)/5.0))
    def b_n(self, V): return 0.25*sym_backend.exp(-(V-20+self.shift)/40.0)

    def z0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+60)/8.5))
    def tz(self, V): return 0.27/sym_backend.exp((V+35.8)/19.7)+sym_backend.exp(-(V+79.7)/12.7)+0.1

    def u0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+78)/6.0))

    #adapted from bazhenov
    def tu(self, V):
        return 0.27/(sym_backend.exp((V+46)/5.0)+sym_backend.exp(-(V+238)/37.5)) \
                    +5.1/2*(1+sym_backend.tanh((V+57)/3))

    def I_Na(self, V, m, h): return self.g_Na_PN*m**3*h*(V - self.E_Na_PN) #nS*mV = pA
    def I_K(self, V, n): return self.g_K_PN*n*(V - self.E_K_PN)
    def I_L(self, V): return self.g_L_PN*(V - self.E_L_PN)
    def I_A(self, V, z, u): return self.g_A_PN*z**4*u*(V - self.E_K_PN)
    def I_KL(self, V): return self.g_KL_PN*(V - self.E_KL_PN)

"Model of Lateral Neurons\
Defined in pico amps"
class LN:
    #Constants for LN

    C_m  =   142.0 # membrane capacitance, in pF
    # maximum conducances, in nS
    g_K_LN  =   1000.0
    g_L_LN  =   21.0
    g_KL_LN =   1.43
    g_Ca_LN =   286.0
    g_KCa_LN=   35.8

    # Nernst reversal potentials, in mV
    E_Na_LN = 50.0
    E_K_LN  = -95.0
    E_L_LN  = -50.0
    E_KL_LN = -95.0
    E_Ca_LN = 140.0

    DIM = 6

    def __init__(self, para = None):
        self.i_inj = 0 # injected currents
        self.V = None
        self.n = None
        self.q = None
        self.s = None
        self.v = None
        self.Ca = None

    def set_integration_index(self, i):
        self.ii = i
        self.V  = y(i)
        self.n  = y(i+1)
        self.s  = y(i+2)
        self.v  = y(i+3)
        self.q  = y(i+4)
        self.Ca = y(i+5)

    def set_neuron_index(self, ni):
        self.ni = ni

    def dydt(self, pre_synapses, pre_neurons): # a list of pre-synaptic neurons
        # define how neurons are coupled here
        VV = self.V
        nn = self.n
        qq = self.q
        ss = self.s
        vv = self.v
        Ca = self.Ca
        i_inj = self.i_inj

        i_syn = sum(self.I_syn(VV, y(synapse.get_ind()),
                    synapse.get_params()[0], synapse.get_params()[1], synapse.weight)
                    for (i,synapse) in enumerate(pre_synapses))

        i_base = (self.I_K_LN(VV, nn) + self.I_L_LN(VV) + self.I_KCa(VV, qq) + \
                        self.I_Ca(VV, ss, vv) + self.I_KL_LN(VV) + i_syn)


        yield -1/self.C_m*(i_base - i_inj)
        yield self.dnl_dt(VV, nn)
        yield self.ds_dt(VV, ss)
        yield self.dv_dt(VV, vv)
        yield self.dq_dt(Ca, qq)
        yield self.dCa_dt(VV, ss, vv, Ca)

    def get_initial_condition(self):
        return [-65.0, 0.0, 0.0, 0.8, 0.0, 0.2]

    def get_ind(self):
        return self.ii

    def get_volt(self):
        return self.V

    def I_syn(self, V, r, gNt, E_nt, w): return gNt*r*w*(V - E_nt)

    def a_nl(self, V): return 0.02*(-(35.0+V)/(sym_backend.exp(-(35.0+V)/5.0)-1.0))
    def b_nl(self, V): return 0.5*sym_backend.exp((-(40.0+V)/40.0))

    def nl0(self, V): return self.a_nl(V)/(self.a_nl(V)+self.b_nl(V))
    def tnl(self, V): return 4.65/(self.a_nl(V)+self.b_nl(V))

    def s0(self, V): return 0.5*(1-sym_backend.tanh(-0.5*(V+20.0)/6.5))
    def ts(self, V): return 1+(V+30)*0.014

    def v0(self, V): return 0.5*(1-sym_backend.tanh(0.5*(V+25.0)/12.0))
    def tv(self, V): return 0.3*sym_backend.exp((V-40)/13.0)+0.002*sym_backend.exp(-(V-60.0)/29.0)

    def q0(self, Ca): return Ca/(Ca+2.0)
    def tq(self, Ca): return 100.0/(Ca+2.0)

    def I_Ca(self, V, s, v): return self.g_Ca_LN*s**2*v*(V-self.E_Ca_LN)
    def I_KCa(self, V, q):   return self.g_KCa_LN*q*(V-self.E_K_LN)
    def I_KL_LN(self, V): return self.g_KL_LN*(V - self.E_KL_LN)
    def I_K_LN(self, V, nl): return  self.g_K_LN*nl**4*(V - self.E_K_LN)
    def I_L_LN(self, V): return self.g_L_LN*(V - self.E_L_LN)


    def dV_dt(self, V, n, q, s, v, t, Ca, i_syn):
        return -1/self.C_m *(self.I_K_LN(V, n) + self.I_L_LN(V) + self.I_KCa(V, q) + \
                        self.I_Ca(V, s, v) + self.I_KL_LN(V) - self.i_inj + i_syn)
    def dCa_dt(self, V, s, v, Ca): return -2.86e-6*self.I_Ca(V, s, v)-(Ca-0.2)/150.0
    def ds_dt(self, V, s): return (self.s0(V)-s)/self.ts(V)
    def dv_dt(self, V, v): return (self.v0(V)-v)/self.tv(V)
    def dq_dt(self, Ca, q): return (self.q0(Ca)-q)/self.tq(Ca)
    def dnl_dt(self, V, nl): return (self.nl0(V)-nl)/self.tnl(V)


class Synapse_gaba_LN:
    #inhibition
    E_gaba = -70.0
    alphaR = 10.0
    betaR = 0.16
    Tm = 1.0


    Kp = 1.5
    Vp = -20.0

    DIM = 1
    def __init__(self, gGABA = 800.0):
        self.r = None
        self.weight = 1.0
        self.gGABA = gGABA

    def set_integration_index(self, i):
        self.ii = i
        self.r = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.weight = w

    def dydt(self, pre_neuron, pos_neuron):
        Vpre = pre_neuron.V
        r = self.r
        yield (self.alphaR*self.Tm/(1+sym_backend.exp(-(Vpre - self.Vp)/self.Kp)))*(1-r) - self.betaR*r

    def get_params(self):
        return [self.gGABA, self.E_gaba]

    def get_initial_condition(self):
        return [0.1]


"""
Different Version of an nAch Synapse
"""
class Synapse_nAch_PN_2:
    #inhibition
    E_nAch = 0.0
    r1 = 1.5
    tau = 1.0
    Kp = 1.5
    Vp = -20.0

    DIM = 1
    def __init__(self, gnAch = 300.0):
        self.r = None
        self.weight = 1.0
        self.gnAch = gnAch


    def set_integration_index(self, i):
        self.ii = i
        self.r = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.weight = w

    def dydt(self, pre_neuron, pos_neuron):
        Vpre = pre_neuron.V
        r = self.r
        yield (self.r_inf(Vpre) - r)/(self.tau*(self.r1-self.r_inf(Vpre)))

    def r_inf(self,V): return 0.5*(1.0-sym_backend.tanh(-0.5*(V - self.Vp)/self.Kp))

    def get_params(self):
        return [self.gnAch, self.E_nAch]

    def get_initial_condition(self):
        return [0.0]

class Synapse_nAch_PN:
    #Excitation
    E_nAch = 0.0
    alphaR = 10.0
    betaR = 0.2
    Tm = 0.5

    Kp = 1.5
    Vp = -20.0

    DIM = 1
    def __init__(self, gnAch = 300.0):
        self.r = None
        self.weight = 1.0
        self.gnAch = gnAch


    def set_integration_index(self, i):
        self.ii = i
        self.r = y(i)

    def get_ind(self):
        return self.ii

    def fix_weight(self, w):
        self.weight = w

    def dydt(self, pre_neuron, pos_neuron):
        Vpre = pre_neuron.V
        r = self.r
        yield (self.alphaR*self.Tm/(1+sym_backend.exp(-(Vpre - self.Vp)/self.Kp)))*(1-r) - self.betaR*r

    def get_params(self):
        return [self.gnAch, self.E_nAch]

    def get_initial_condition(self):
        return [0.0]
