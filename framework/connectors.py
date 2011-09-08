# encoding: utf-8
import numpy
import pylab
from pylab import griddata
from interfaces import MozaikComponent
from sheets import SheetWithMagnificationFactor
from interfaces import VisualSystemConnector
import logging
from NeuroTools import visual_logging
from pyNN.common import Population
from NeuroTools.parameters import ParameterSet, ParameterDist
from pyNN import random, space
from MozaikLite.tools.misc import sample_from_bin_distribution

class MozaikLiteVisualSystemConnector(VisualSystemConnector):
      required_parameters = ParameterSet({
      'target_synapses' : str,
      })
      
      def __init__(self, network, source, target, parameters):
          VisualSystemConnector.__init__(self, network, source,target,parameters)

    
      def connection_field_plot_continuous(self,index,weights,afferent=True,density=30):
          
          #!HACKALERT
          # this is only for compatibility with original mozaik retinas
          # should be deleted once everything is unified
          if not isinstance(source,Population):
             source = source.pop
          
          weights =  self.proj.getWeights(format='array')
          print 'max:',numpy.max(weights)
          print 'min:', numpy.min(weights)
          x = []
          y = []
          w = []
          
          if afferent:
                 weights = weights[:,index].ravel()
                 p = self.proj.pre
          else:
                 weights = weights[index,:].ravel()
                 p = self.proj.post
          
          for (ww,i) in zip(weights,numpy.arange(0,len(weights),1)):
                    x.append(p.positions[0][i])
                    y.append(p.positions[1][i])
                    w.append(ww)
              
          bx = min([min(p.positions[0]),min(p.positions[0])])  
          by = max([max(p.positions[1]),max(p.positions[1])])  
          xi = numpy.linspace(min(p.positions[0]),max(p.positions[0]),100)
          yi = numpy.linspace(min(p.positions[1]),max(p.positions[1]),100)
          zi = griddata(x,y,w,xi,yi)
          pylab.figure()
          pylab.imshow(zi)
          pylab.colorbar()
          #pylab.title('Connection field from %s to %s of neuron %d' % (self.proj.pre.name,self.proj.post.name,index))

      def connection_field_plot_scatter(self,index,weights,afferent=True,density=30):
          
          weights =  self.proj.getWeights(format='array')
          
          print 'max:',numpy.max(weights)
          print 'min:', numpy.min(weights)
          x = []
          y = []
          w = []
          print index
          print numpy.shape(weights)
          if afferent:
                 weights = weights[:,index].ravel()
                 p = self.proj.pre
          else:
                 weights = weights[index,:].ravel()
                 p = self.proj.post
          
          for (ww,i) in zip(weights,numpy.arange(0,len(weights),1)):
                    x.append(p.positions[0][i])
                    y.append(p.positions[1][i])
                    w.append(ww)
          
          import pylab    
          pylab.figure()
          pylab.scatter(x,y,c=w,s=20)
          pylab.xlim(min(p.positions[0]),max(p.positions[0]))
          pylab.ylim(min(p.positions[1]),max(p.positions[1]))
          pylab.colorbar()
          #pylab.show()
          #pylab.title('Connection field from %s to %s of neuron %d' % (self.proj.pre.name,self.proj.post.name,index))


class ExponentialProbabilisticArborization(MozaikLiteVisualSystemConnector):
    required_parameters = ParameterSet({
        'weights': float,                #nA, the cell type of the sheet 
        'propagaion_constant': float,    #ms/μm the constant that will determinine the distance dependent delays on the connections
        'arborization_constant': float,  # μm distance constant of the exponential decay of the probability of the connection with respect
                                         # to the distance from the invervation point.
        'synapse_dynamics' : str,        # string indetifying the synaptic plasticity mechanism (None for no plasticity)
    })


    def __init__(self, network, source, target, parameters,name):
        MozaikLiteVisualSystemConnector.__init__(self, network, source,target,parameters)
        
        self.name = name
        
        if isinstance(target, SheetWithMagnificationFactor):
            dist = "target.dvf_2_dcs(d)"
        else:
            dist = "d"
    		
        if parameters.synapse_dynamics == 'None':
            synapse_dynamics = None
        else:
	        synapse_dynamics = parameters.synapse_dynamics
	

        arborization_expression = "exp(-abs(("+ dist + ")/" + str(parameters.arborization_spread) + "))"
        delay_expression = dist + "*" + parameters.propagation_constant 

        method = self.sim.DistanceDependentProbabilityConnector(arborization_expression,allow_self_connections=False, weights=parameters.weights, delays=delay_expression, space=space.Space(axes='xy'), safe=True, verbose=False, n_connections=None)

        self.proj = self.sim.Projection(source.pop, target.pop, method, synapse_dynamics=parameters.synapse_dynamics, label=self.name, rng=None, target=parameters.target_synapses)
        
class UniformProbabilisticArborization(MozaikLiteVisualSystemConnector):

        required_parameters = ParameterSet({
            'connection_probability': float, #probability of connection between two neurons from the two populations
            'weights': float,                #nA, the cell type of the sheet 
            'propagation_constant': float,    #ms/μm the constant that will determinine the distance dependent delays on the connections
            'synapse_dynamics' : str,        # string indetifying the synaptic plasticity mechanism (None for no plasticity)
        })


        def __init__(self, network, source, target, parameters,name):
            MozaikLiteVisualSystemConnector.__init__(self, network, source,target,parameters)
            self.name = name
            if parameters.synapse_dynamics == 'None':
                synapse_dynamics = None
            else:
                synapse_dynamics = parameters.synapse_dynamics
	        
            method = self.sim.FixedProbabilityConnector(parameters.connection_probability,allow_self_connections=False, weights=parameters.weights, delays=parameters.propagation_constant, space=space.Space(axes='xy'), safe=True)
            self.proj = self.sim.Projection(source.pop, target.pop, method, synapse_dynamics=synapse_dynamics, label=self.name, rng=None, target=parameters.target_synapses)

class SpecificArborization(MozaikLiteVisualSystemConnector):
	  """
	  Generic connector which gets directly list of connections as the list of quadruplets as accepted by the
	  pyNN FromListConnector.
	  
	  This connector cannot be parametrized directly via the parameter file because that does not suport list of tuples.
	  """
	  
	  required_parameters = ParameterSet({
	    'synapse_dynamics' : str, # string indetifying the synaptic plasticity mechanism (None for no plasticity)
        'weight_factor': float, # weight scaler
	  })
	  
	  def __init__(self, network, source, target, connection_list,parameters,name):
    	    MozaikLiteVisualSystemConnector.__init__(self, network, source,target,parameters)
            self.name = name
            
            connection_list = [(a,b,c*parameters.weight_factor,d) for (a,b,c,d) in connection_list]

            method  =  self.sim.FromListConnector(connection_list)
            
            if parameters.synapse_dynamics == 'None':
                synapse_dynamics = None
            else:
                synapse_dynamics = parameters.synapse_dynamics

	        
            #!HACKALERT
            # this is only for compatibility with original mozaik retinas
            # should be deleted once everything is unified
            if not isinstance(source,Population):
                source = source.pop
            self.proj = self.sim.Projection(source, target.pop, method, synapse_dynamics=synapse_dynamics, label=self.name, rng=None, target=parameters.target_synapses)

class SpecificProbabilisticArborization(MozaikLiteVisualSystemConnector):
	  """
	  Generic connector which gets directly list of connections as the list of quadruplets as accepted by the
	  pyNN FromListConnector.
	  
	  This connector cannot be parametrized directly via the parameter file because that does not suport list of tuples.
	  """
	  
	  required_parameters = ParameterSet({
	    'synapse_dynamics' : str,        # string indetifying the synaptic plasticity mechanism (None for no plasticity)
        'weight_factor': float, # the base size of weights
        'num_samples' : int
	  })
	  
	  def __init__(self, network, source, target, connection_list,parameters,name):
    	    MozaikLiteVisualSystemConnector.__init__(self, network, source,target,parameters)
            self.name = name
            
            if parameters.synapse_dynamics == 'None':
                synapse_dynamics = None
            else:
                synapse_dynamics = parameters.synapse_dynamics

            samples = sample_from_bin_distribution([c[2] for c in connection_list],parameters.num_samples)
            
            connection_list = [connection_list[s] for s in samples]
            connection_list = [(a,b,parameters.weight_factor,d) for (a,b,c,d) in connection_list]

            method  =  self.sim.FromListConnector(connection_list)  
            
            #!HACKALERT
            # this is only for compatibility with original mozaik retinas
            # should be deleted once everything is unified
            if not isinstance(source,Population):
                source = source.pop
            self.proj = self.sim.Projection(source, target.pop, method, synapse_dynamics=synapse_dynamics, label=self.name, rng=None, target=parameters.target_synapses)



def gabor(x1,y1,x2,y2,orientation,frequency,phase,size,aspect_ratio):
        from numpy import cos,sin
        X = (x1-x2)*cos(orientation) + (y1-y2)*sin(orientation)
        Y = -(x1-x2)*sin(orientation) + (y1-y2)*cos(orientation)
        ker = - (X*X + Y*Y*(aspect_ratio**2)) / (2*(size**2))
        return numpy.exp(ker)*numpy.cos(2*numpy.pi*X*frequency+phase)




class GaborConnector(MozaikComponent):
      required_parameters = ParameterSet({
          'aspect_ratio': ParameterDist, #aspect ratio of the gabor
          'size':         ParameterDist, #the size of the gabor  RFs in degrees of visual field
          'orientation':  ParameterDist, #the orientation of the gabor RFs
          'phase':        ParameterDist, #the phase of the gabor RFs
          'frequency':    ParameterDist, #the frequency of the gabor in degrees of visual field 
          
          'topological': bool, # should the receptive field centers vary with the position of the given neurons 
          'probabilistic': bool, # should the weights be probabilistic or directly proportianal to the gabor profile
          'propagation_constant': float,    #ms/μm the constant that will determinine the distance dependent delays on the connections
          
          'specific_arborization' : ParameterSet,
        
      })
	    

      def __init__(self, network, lgn_on, lgn_off, target, parameters,name):
    	     MozaikComponent.__init__(self, network,parameters)
             self.name = name
             
             # this is only for compatibility with original mozaik retinas
             # should be deleted once everything is unified
             if isinstance(lgn_on,Population):
                on = lgn_on
             else:
                on = on.pop

             if isinstance(lgn_off,Population):
                off = lgn_off
             else:
                off = off.pop
             
             on_weights=[] 
             off_weights=[]
             z = []
             z =1
             
             for (neuron1,i) in zip(on,numpy.arange(0,len(on),1)):
                for (neuron2,j) in zip(target.pop,numpy.arange(0,len(target.pop),1)):
                    
                    orientation = parameters.orientation.next()[0]
                    aspect_ratio = parameters.aspect_ratio.next()[0]
                    frequency = parameters.frequency.next()[0]
                    size = parameters.size.next()[0]
                    phase = parameters.phase.next()[0]
                    if parameters.topological:
                        on_weights.append((i,j,numpy.max((0,gabor(on.positions[0][i],on.positions[1][i],target.pop.positions[0][j],target.pop.positions[1][j],orientation,frequency,phase,size,aspect_ratio))),parameters.propagation_constant))
                        off_weights.append((i,j,-numpy.min((0,gabor(off.positions[0][i],off.positions[1][i],target.pop.positions[0][j],target.pop.positions[1][j],orientation,frequency,phase,size,aspect_ratio))),parameters.propagation_constant))
                    else:
                        on_weights.append((i,j,numpy.max((0,gabor(on.positions[0][i],on.positions[1][i],0,0,orientation,frequency,phase,size,aspect_ratio))),parameters.propagation_constant))
                        off_weights.append((i,j,-numpy.min((0,gabor(off.positions[0][i],off.positions[1][i],0,0,orientation,frequency,phase,size,aspect_ratio))),parameters.propagation_constant))
             if parameters.probabilistic:
                 on_proj =  SpecificProbabilisticArborization(network,lgn_on,target,on_weights,parameters.specific_arborization,'ON_to_[' + target.name + ']')
                 off_proj = SpecificProbabilisticArborization(network,lgn_off,target,off_weights,parameters.specific_arborization,'OFF_to_[' + target.name + ']')
             else:
                 on_proj =  SpecificArborization(network,lgn_on,target,on_weights,parameters.specific_arborization,'ON_to_[' + target.name + ']')
                 off_proj = SpecificArborization(network,lgn_off,target,off_weights,parameters.specific_arborization,'OFF_to_[' + target.name + ']')
                 
                 
             on_proj.connection_field_plot_scatter(len(target.pop)-1,on_weights)   
             off_proj.connection_field_plot_scatter(len(target.pop)-1,off_weights)   
