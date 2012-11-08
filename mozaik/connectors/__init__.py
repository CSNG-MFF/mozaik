# encoding: utf-8
"""
Mozaik connector interface.
"""
import math
import numpy
import pylab
import mozaik
from pylab import griddata
from mozaik.framework.interfaces import Connector
from mozaik.framework.sheets import SheetWithMagnificationFactor
from NeuroTools.parameters import ParameterSet, ParameterDist
from mozaik.tools.misc import sample_from_bin_distribution, normal_function
from collections import Counter
from pyNN import random, space

logger = mozaik.getMozaikLogger("Mozaik")


class MozaikConnector(Connector):

    required_parameters = ParameterSet({
            'target_synapses' : str,
            'short_term_plasticity': ParameterSet({
                    'u': float, 
                    'tau_rec': float, 
                    'tau_fac': float,
                    'tau_psc': float
            }),
    
    })
    
    def __init__(self, network, name,source, target, parameters):
      Connector.__init__(self, network, name, source,target,parameters)
      
      if not self.parameters.short_term_plasticity != None:
        self.short_term_plasticity = None
      else:
        #self.short_term_plasticity = self.sim.SynapseDynamics(fast=self.sim.TsodyksMarkramMechanism(**self.parameters.short_term_plasticity_params))                    
        self.short_term_plasticity = self.sim.NativeSynapseDynamics("tsodyks_synapse", self.parameters.short_term_plasticity)
        
    def connect(self):
      raise NotImplementedError

    def connection_field_plot_continuous(self, index, afferent=True, density=30):
        weights = self.proj.getWeights(format='array')
        x = []
        y = []
        w = []
        
        if afferent:
            weights = weights[:, index].ravel()
            p = self.proj.pre
        else:
            weights = weights[index, :].ravel()
            p = self.proj.post

        for (ww, i) in zip(weights, numpy.arange(0, len(weights), 1)):
                if not math.isnan(ww):                
                    x.append(p.positions[0][i])
                    y.append(p.positions[1][i])
                    w.append(ww)
                        
        xi = numpy.linspace(min(x), max(x), 100)
        yi = numpy.linspace(min(y), max(y), 100)
        zi = griddata(x, y, w, xi, yi)
        pylab.figure()
        #pylab.imshow(zi)
        pylab.scatter(x,y,marker='o',c=w,s=5,zorder=10)
        pylab.title('Connection field from %s to %s of neuron %d' % (self.source.name,
                                                                     self.target.name,
                                                                     index))
        #pylab.colorbar()

    def store_connections(self, datastore):
        from mozaik.analysis.analysis_data_structures import Connections
        weights = numpy.nan_to_num(self.proj.getWeights(format='array'))
        datastore.add_analysis_result(
            Connections(weights,
                        name=self.name,
                        source_name=self.source.name,
                        target_name=self.target.name,
                        analysis_algorithm=self.__class__.__name__))


class SpecificArborization(MozaikConnector):
    """
    Generic connector which gets directly list of connections as the list of
    quadruplets as accepted by the pyNN FromListConnector.

    This connector cannot be parametrized directly via the parameter file
    because that does not support list of tuples.
    """

    required_parameters = ParameterSet({
        'weight_factor': float,  # weight scaler
    })

    def __init__(self, network, source, target, connection_list, parameters, name):
        MozaikConnector.__init__(self, network, name, source,
                                             target, parameters)
        self.connection_list = connection_list

    def connect(self):
        self.connection_list = [(a, b, c*self.parameters.weight_factor, d)
                                for (a, b, c, d) in self.connection_list]
        self.method = self.sim.FromListConnector(self.connection_list)
        self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                self.method,
                                synapse_dynamics=self.short_term_plasticity,
                                label=self.name,
                                rng=None,
                                target=self.parameters.target_synapses)


class SpecificProbabilisticArborization(MozaikConnector):
    """
    Generic connector which gets directly list of connections as the list
    of quadruplets as accepted by the pyNN FromListConnector.

    It interprets the weights as proportional probabilities of connectivity,
    and for each neuron out connections it samples num_samples of
    connections that actually get realized according to these weights.
    Each such sample connections will have weight equal to
    weight_factor/num_samples but note that there can be multiple
    connections between a pair of neurons in this sample (in which case the
    weights are set to the multiple of the base weights times the number of
    occurrences in the sample).

    This connector cannot be parameterized directly via the parameter file
    because that does not support list of tuples.
    """

    required_parameters = ParameterSet({
        'weight_factor': float,  # the overall strength of synapses in this connection per neuron (in µS) (i.e. the sum of the strength of synapses in this connection per target neuron)
        'num_samples': int
    })

    def __init__(self, network, source, target, connection_list, parameters, name):
        MozaikConnector.__init__(self, network, name, source,
                                             target, parameters)
        self.connection_list = connection_list

    def connect(self):
        cl = []
        d = {}

        for i, (s, t, w, delay) in enumerate(self.connection_list):
            if t in d:
                d[t].append(i)
            else:
                d[t] = [i]

        for k in d:
            w = [self.connection_list[i][2] for i in d[k]]
            samples = sample_from_bin_distribution(w, self.parameters.num_samples)
            a = numpy.array([self.connection_list[d[k][s]]
                             for s in samples])[:, [0, 1, 3]]
            z = Counter([tuple(z) for z in a.tolist()])

            cl.extend([(a, b,
                        self.parameters.weight_factor/len(samples) * z[(a, b, de)],
                        de)
                      for (a, b, de) in z.keys()])

        method = self.sim.FromListConnector(cl)
        self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                method,
                                synapse_dynamics=self.short_term_plasticity,
                                label=self.name,
                                rng=None,
                                target=self.parameters.target_synapses)
                  


