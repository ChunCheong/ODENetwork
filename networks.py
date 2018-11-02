"""
networks.py
A module that contains all the networks architecture classes.
Define layers or combination of layers here.
"""
import numpy as np
import neuron_models as nm

class Network:
    """
    Base class for all networks. Directed graph by default.
    Args:
        vertexs: list of neurons
        adja_list: list of lists of pre-synaptic neuron indices
        edges: list of lists of pre-synapses
    Example:
    Consider a network with 3 neurons; both neruon 0 and 1 synapse ONTO 2.
    vertexs = [neruon0, neruon1, neruon2]
    adja_list = [[],[],[0,1]]
    edges = [[],[],[synapse0, synapse1]]
    i.e. In the above network, synapse0 is the one connects 0 to 2, and synapse1
    connects 1 to 2.

    The orders of the elements of the three lists have to match.
    i.e. adja_list[n] = list_of_neruons_that_synapse_onto_neuron_n
    edges[n] = list_of_pre_synapses_onto_neuron_n
    """
    def __init__(self,vertexs,adja_list,edges_list=None):
        # quick check
        if len(vertexs) != len(adja_list):
            raise ValueError('vertexs does not match adja_list')
        num_bonds = sum(len(pre_syn_neurons) for pre_syn_neurons in adja_list)
        if (edges_list is not None):
            num_edges = sum(len(pre_synapses) for pre_synapses in edges_list)
            if num_bonds != num_edges:
                raise ValueError('adja_list does not match edges_list')
        self.vertexs = vertexs
        self.adja_list = adja_list
        self.edges_list = edges_list

class FeedForwardFC2Layer(Network):
    """
    A fully connected network with 2 layers.
    layer1(2) is the list of neurons in layer 1(2).
    Convention: layer1 synaspes ONTO layer2.

    Again, edge's order is defined to be such that it matches the order of
    appearance in adja_list. See class Network.
    """
    def __init__(self,layer_1,layer_2,edges_list=None):
        # quick check
        num_1, num_2 = len(layer_1), len(layer_2)
        if (edges_list is not None):
            num_edges = sum(len(pre_synapses) for pre_synapses in edges_list)
            if (num_edges != num_1*num_2):
                raise ValueError('Number of edges does not match as FC layer')
        # Define network topology via adjacency list and implicitly define the
        # order of neurons.
        adja_list = [ [] for j in range(num_1)] # no synapses onto layer1
        adja_list += [
            [i for i in range(num_1)] for j in range(num_1, num_1+num_2)]
        # Populate the network
        vertexs = layer_1 + layer_2
        Network.__init__(self, vertexs, adja_list, edges_list)
        self.layer_1 = layer_1
        self.layer_2 = layer_2

class HHSTDPFeedForwardFC2Layer(FeedForwardFC2Layer):
    """
    A fully connected network with 2 layers made of HH neruons, with dynamical
    synapses with calcium-based STDP.
    Again, syanpses's order is defined to be such that it matches the order of
    appearance in adja_list. See class Network.
    """
    def __init__(self, num_1, num_2):
        # Populate first layer
        neuron = nm.HHNeuronWithCa
        layer_1 = [neuron() for n in range(num_1)]
        # Up to some minor modifications we can repeat this for several groups
        # if needed, e.g.
        # neuron = nm.AnotherNeuronType
        # dim = neuron_type.dim
        # num_neuron = some_number
        # populate second group
        # neurons_layer_1 += [neuron(n,i+n*dim) for n in range(num_neuron)]
        # i += num_neuron*dim
        # n0 += num_neuron
        # ......
        # and repeat as many times as needed.
        # Populate second layer
        neuron = nm.HHNeuronWithCa
        layer_2 = [neuron() for n in range(num_2)]
        # Populate synapses (for each layer)
        # Important note:
        # We should populate synapses according to the order induced by the
        # adjacency list, especially when there are mutiple kind of synapses.
        # See class Network.
        synapse = nm.PlasticNMDASynapse
        num_synapses = num_1*num_2 # fully connected
        synapse_list = []
        FeedForwardFC2Layer.__init__(self, layer_1, layer_2)
        for (n, neighbors) in enumerate(self.adja_list): # inherit from FeedForwardFC2Layer
            num_neigbors = len(neighbors)
            pre_synapses = [synapse() for s in range(num_neigbors)]
            synapse_list.append(pre_synapses)
        self.edges_list = synapse_list

# # Some obsolete codes for historical purposes:
#
# # def get_n2i_s2i_dim(neurons_layer_1, neruon_dims, synapse_nums, synapse_dims):
# #     """Return the n2i and s2i arrays.
# #
# #     To obtain all the ODEs, we need to assign integration varibale index i to
# #     each neuron's (synapse's) varibale. We do so by making two arrays for
# #     book-keeping. The first one gives integration index i at position n, where
# #     n is neuron index, while the second one gives i at position s, where
# #     s is synapse index.
# #
# #     Examples:
# #     neuron_nums, neruon_dims = [3], [2]
# #     synapse_nums, synapse_dims = [5], [7]
# #     n2i, s2i, dim_tot = get_n2i_s2i(
# #         neuron_nums, neruon_dims, synapse_nums, synapse_dims)
# #     gives
# #     n2i = array([0, 2, 4])
# #     s2i = array([ 6, 13, 20, 27, 34])
# #     dim_tot = 41
# #
# #     Args:
# #         neuron_nums: list of (integer) numbers of neruons of a sepecific kind.
# #         neruon_dims: list of (integer) dimensions of neruons.
# #         synapse_nums: list of (integer) numbers of synapses of a sepecific kind.
# #         synapse_dims: list of (integer) dimensions of synaspes.
# #     """
# #
# #     # Preallocate the return array
# #     neruon_num_tot = sum(neuron_nums)
# #     n2i = np.empty(neruon_num_tot, dtype=int)
# #     # First index neurons
# #     i = 0
# #     for (k, neuron_num) in enumerate(neuron_nums):
# #         neuron_dim = neruon_dims[k]
# #         n2i[k:(k+neuron_num)] = np.arange(k,(k+neuron_num))*neuron_dim + i
# #         i += neuron_num*neuron_dim
# #     # Then populate synapses
# #     syanpse_num_tot = sum(synapse_nums)
# #     s2i = np.empty(syanpse_num_tot, dtype=int)
# #     for (k, synapse_num) in enumerate(synapse_nums):
# #         synapse_dim = synapse_dims[k]
# #         s2i[k:(k+synapse_num)] = np.arange(k,(k+synapse_num))*synapse_dim + i
# #         i += synapse_num*synapse_dim
# #     dim_tot = i
# #     return n2i, s2i, dim_tot
