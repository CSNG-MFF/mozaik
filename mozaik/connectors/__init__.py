# encoding: utf-8
"""
Mozaik connector interface.
"""
import math
import numpy
import pylab
import mozaik
import time
from pylab import griddata
from mozaik.framework.interfaces import Connector
from mozaik.framework.sheets import SheetWithMagnificationFactor
from NeuroTools.parameters import ParameterSet, ParameterDist
from mozaik.tools.misc import sample_from_bin_distribution, normal_function
from collections import Counter
from pyNN import random, space

logger = mozaik.getMozaikLogger("Mozaik")


class MozaikConnector(Connector):
    """
    An abstract interface class for Connectors in mozaik. Each mozaik connector should derive from this class and implement 
    the _connect function. The usage is: create the instance of MozaikConnector and call connect() to realize the connections.
    """
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
          t0 = time.time()
          self._connect()
          connect_time = time.time() - t0
          logger.info('Connector %s took %.0fs to compute' % (self.__class__.__name__,connect_time))
            
        
    def _connect(self):
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

    def __init__(self, network, source, target, connection_matrix,delay_matrix, parameters, name):
        MozaikConnector.__init__(self, network, name, source,
                                             target, parameters)
        self.connection_matrix = connection_matrix
        self.delay_matrix = delay_matrix

    def _connect(self):
        X,Y = numpy.meshgrid(numpy.arange(0,self.source.pop.size,1),numpy.arange(0,self.target.pop.size,1))
        self.connection_list = zip((X.flatten(),Y.flatten(),self.connection_matrix.flatten()*self.parameters.weight_factor,self.delay_matrix.flatten()))
        
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

    def __init__(self, network, source, target, connection_matrix,delay_matrix, parameters, name):
        MozaikConnector.__init__(self, network, name, source,target, parameters)
        self.connection_matrix = connection_matrix
        self.delay_matrix = delay_matrix

    def _connect(self):
        X,Y = numpy.meshgrid(numpy.arange(0,self.source.pop.size,1),numpy.arange(0,self.target.pop.size,1))
        weights = self.connection_matrix
        delays = self.delay_matrix
        
        cl = []
        
        for i in xrange(0,self.target.pop.size):
            co = Counter(sample_from_bin_distribution(weights[:,i].flatten(), int(self.parameters.num_samples)))
            cl.extend([(k,i,self.parameters.weight_factor*co[k]/self.parameters.num_samples,delays[k][i]) for k in co.keys()])
            
        method = self.sim.FromListConnector(cl)
        self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                method,
                                synapse_dynamics=self.short_term_plasticity,
                                label=self.name,
                                rng=None,
                                target=self.parameters.target_synapses)
                  


