"""
neuron_models.py
A module that contains all the neuron and synapse model classes.
To-dos:
1. make method/function to check dimensions consistency across DIM, ic,...etc
2. introduce delay, related to choices for rho_gate placement
3. too many "self"... can improve readability?
"""
from jitcode import jitcode, y, t
import numpy as np
import symengine

# "Global" constants, if any

# Some very common helper functions
def sigmoid(x):
    return 1./(1.+ symengine.exp(-x))

def heaviside(x):
    K = 1e3 # some big number
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
    TAU_W = 100.
    # stdp stuff
    THETA_P = 1.3
    THETA_D = 1.
    GAMMA_P = 22
    GAMMA_D = 10
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
    AVO_CONST = 0.03 # "Avogadros" constant, relate calcium concentraion and current

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
        return tau_x_0 + tau_x_1*(1-(symengine.tanh((Vm - V_0)/sigma_x))**2)

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
        return - self.COND_CA_SYN*W_ij*rho_ij*(Vm_po - self.RE_PO_CA)
