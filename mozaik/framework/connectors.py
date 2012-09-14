# encoding: utf-8
from collections import Counter
import numpy
import pylab
import mozaik
from pylab import griddata
from interfaces import MozaikComponent
from sheets import SheetWithMagnificationFactor
from interfaces import Connector
from NeuroTools.parameters import ParameterSet, ParameterDist
from pyNN import random, space
from mozaik.tools.misc import sample_from_bin_distribution, normal_function
from mozaik.tools.circ_stat import circular_dist,circ_mean
from scipy.interpolate import NearestNDInterpolator

logger = mozaik.getMozaikLogger("Mozaik")

class mozaikVisualSystemConnector(Connector):
      required_parameters = ParameterSet({
                'target_synapses' : str,
                'short_term_plasticity' : bool,
                'short_term_plasticity_params': ParameterSet({
                        'U': float, 
                        'tau_rec': float, 
                        'tau_facil': float
                }),

      })
      
      def __init__(self, network, name,source, target, parameters):
          Connector.__init__(self, network, name, source,target,parameters)
          
          if not self.parameters.short_term_plasticity:
            self.short_term_plasticity = None
          else:
            self.short_term_plasticity = self.sim.SynapseDynamics(fast=self.sim.TsodyksMarkramMechanism(**self.parameters.short_term_plasticity_params))                    
    
      def connect(self):
        raise NotImplementedError
        pass
    
      def connection_field_plot_continuous(self,index,afferent=True,density=30):
          weights =  self.proj.getWeights(format='array')
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
          pylab.title('Connection field from %s to %s of neuron %d' % (self.source.name,self.target.name,index))
          pylab.colorbar()
          

      def store_connections(self,datastore):
          from mozaik.analysis.analysis_data_structures import Connections
          weights =  numpy.nan_to_num(self.proj.getWeights(format='array'))
          datastore.add_analysis_result(Connections(weights,name=self.name,source_name=self.source.name,target_name=self.target.name,analysis_algorithm=self.__class__.__name__))

class ExponentialProbabilisticArborization(mozaikVisualSystemConnector):
    required_parameters = ParameterSet({
        'weights': float,                # nA, the synapse strength 
        'propagation_constant': float,   # ms/μm the constant that will determinine the distance dependent delays on the connections
        'arborization_constant': float,  # μm distance constant of the exponential decay of the probability of the connection with respect (in cortical distance)
                                         # to the distance from the innervation point.
        'arborization_scaler': float,    # the scaler of the exponential decay
    })


    def __init__(self, network, source, target, parameters,name):
        mozaikVisualSystemConnector.__init__(self, network, name,source,target,parameters)
	
    def connect(self):
        # JAHACK, 0.1 as minimal delay should be replaced with the simulations time_step
        if isinstance(self.target, SheetWithMagnificationFactor):
            self.arborization_expression = lambda d: self.parameters.arborization_scaler*numpy.exp(-0.5*(self.target.dvf_2_dcs(d)/self.parameters.arborization_constant)**2)/(self.parameters.arborization_constant*numpy.sqrt(2*numpy.pi))
            self.delay_expression = lambda d:  numpy.maximum(self.target.dvf_2_dcs(d) * self.parameters.propagation_constant,0.1)
        else:
            self.arborization_expression = lambda d: self.parameters.arborization_scaler*numpy.exp(-0.5*(d/self.parameters.arborization_constant)**2)/(self.parameters.arborization_constant*numpy.sqrt(2*numpy.pi))
            self.delay_expression = lambda d:  numpy.maximum(d * self.parameters.propagation_constant,0.1)
        method = self.sim.DistanceDependentProbabilityConnector(self.arborization_expression,allow_self_connections=False, weights=self.parameters.weights, delays=self.delay_expression, space=space.Space(axes='xy'), safe=True, verbose=False, n_connections=None)
        self.proj = self.sim.Projection(self.source.pop, self.target.pop, method, synapse_dynamics=self.short_term_plasticity, label=self.name, rng=None, target=self.parameters.target_synapses)
        
class UniformProbabilisticArborization(mozaikVisualSystemConnector):

        required_parameters = ParameterSet({
            'connection_probability': float, #probability of connection between two neurons from the two populations
            'weights': float,                #nA, the synapse strength 
            'delay': float,    #ms delay of the connections
        })


        def __init__(self, network, source, target, parameters,name):
            mozaikVisualSystemConnector.__init__(self, network, name,source,target,parameters)
	
        def connect(self):
            method = self.sim.FixedProbabilityConnector(self.parameters.connection_probability,allow_self_connections=False, weights=self.parameters.weights, delays=self.parameters.delay, space=space.Space(axes='xy'), safe=True)
            self.proj = self.sim.Projection(self.source.pop, self.target.pop, method, synapse_dynamics=self.short_term_plasticity, label=self.name, rng=None, target=self.parameters.target_synapses)

class SpecificArborization(mozaikVisualSystemConnector):
        """
        Generic connector which gets directly list of connections as the list of quadruplets as accepted by the
        pyNN FromListConnector.
        
        This connector cannot be parametrized directly via the parameter file because that does not suport list of tuples.
        """
		
        required_parameters = ParameterSet({
            'weight_factor': float, # weight scaler
        })
            
        def __init__(self, network, source, target, connection_list,parameters,name):
            mozaikVisualSystemConnector.__init__(self, network, name,source,target,parameters)
            self.connection_list = connection_list    

        def connect(self):	        
            self.connection_list = [(a,b,c*self.parameters.weight_factor,d) for (a,b,c,d) in self.connection_list]
            self.method  =  self.sim.FromListConnector(self.connection_list)
            self.proj = self.sim.Projection(self.source.pop, self.target.pop, self.method, synapse_dynamics=self.short_term_plasticity, label=self.name, rng=None, target=self.parameters.target_synapses)

class SpecificProbabilisticArborization(mozaikVisualSystemConnector):
        """
        Generic connector which gets directly list of connections as the list of quadruplets as accepted by the
        pyNN FromListConnector.
        
        It interprets the weights as proportianal probabilities of connectivity, and for each neuron
        out connections it samples num_samples of connections that actually get realized according to 
        these weights. Each such sample connections will have weight equal to weight_factor/num_samples
        but note that there can be multiple connections between a pair of neurons in this sample 
        (in which case the weights is set to the multiple of the base weights times the number of
        occurances in the sample).
        
        This connector cannot be parametrized directly via the parameter file because that does not suport list of tuples.
        """
        
        required_parameters = ParameterSet({
            'weight_factor': float, # the overall strength of synapses in this connection per neuron (in µS) (i.e. the sum of the strength of synapses in this connection per target neuron)
            'num_samples' : int
        })
        
        def __init__(self, network, source, target, connection_list,parameters,name):
            mozaikVisualSystemConnector.__init__(self, network, name,source,target,parameters)
            self.connection_list = connection_list    
                
        def connect(self):
            cl = []
            d={}
            
            for i,(s,t,w,delay) in enumerate(self.connection_list):
                if d.has_key(t):
                   d[t].append(i)
                else:
                   d[t] = [i]
            
            for k in d.keys():
                w = [self.connection_list[i][2] for i in d[k]]
                samples = sample_from_bin_distribution(w,self.parameters.num_samples)
                a = numpy.array([self.connection_list[d[k][s]] for s in samples])[:,[0,1,3]]
                z = Counter([tuple(z) for z in a.tolist()])
                
                cl.extend([(a,b,self.parameters.weight_factor/len(samples)*z[(a,b,de)],de) for (a,b,de) in z.keys()])
            
            method  =  self.sim.FromListConnector(cl)  
            self.proj = self.sim.Projection(self.source.pop, self.target.pop, method, synapse_dynamics=self.short_term_plasticity, label=self.name, rng=None, target=self.parameters.target_synapses)


class V1PushPullProbabilisticArborization(MozaikComponent):
        """
        This connector implements the standard V1 functionally specific connection rule:
        
        Excitatory synapses are more likely on cooriented in-phase neurons
        Inhibitory synapses are more likely to cooriented anti-phase neurons
        """
        
        required_parameters = ParameterSet({
            'probabilistic': bool, # should the weights be probabilistic or directly proportianal to the gabor profile		
            'or_sigma' : float, # how sharply does the probability of connection fall of with orientation difference
            'propagation_constant': float,    #ms/μm the constant that will determinine the distance dependent delays on the connections
            'phase_sigma' : float, # how sharply does the probability of connection fall of with phase difference
            'specific_arborization' : ParameterSet,
        })
        
        def __init__(self, network, source, target, parameters,name):
            MozaikComponent.__init__(self,network,parameters)
            self.name = name
            self.source = source
            self.target = target
            weights = []
            
            
            for (i,neuron1) in enumerate(self.target.pop.all()):
                for (j,neuron2) in enumerate(self.source.pop.all()):
            
                    or_dist = circular_dist(self.target.get_neuron_annotation(i,'LGNAfferentOrientation'),self.source.get_neuron_annotation(j,'LGNAfferentOrientation'),numpy.pi) / (numpy.pi/2)
                    
                    if self.parameters.specific_arborization.target_synapses == 'excitatory':
                            phase_dist = circular_dist(self.target.get_neuron_annotation(i,'LGNAfferentPhase'),self.source.get_neuron_annotation(j,'LGNAfferentPhase'),2*numpy.pi) / numpy.pi
                    elif self.parameters.specific_arborization.target_synapses == 'inhibitory':
                            phase_dist = (numpy.pi - circular_dist(self.target.get_neuron_annotation(i,'LGNAfferentPhase'),self.source.get_neuron_annotation(j,'LGNAfferentPhase'),2*numpy.pi)) / numpy.pi
                    else:
                        logger.error('Unknown type of synapse!')
                        return	
            
                    or_gauss = normal_function(or_dist,mean=0,sigma=self.parameters.or_sigma)
                    phase_gauss = normal_function(phase_dist,mean=0,sigma=self.parameters.phase_sigma)
                    w = phase_gauss*or_gauss
                    
                    weights.append((j,i,w,self.parameters.propagation_constant))
            
            #we = numpy.zeros((len(self.source.pop),len(self.target.pop)))
            
            
            pnv_source = []
            pnv_target = []
            for (i,neuron1) in enumerate(self.target.pop.all()):
                pnv_target.append(self.target.get_neuron_annotation(i,'LGNAfferentOrientation'))
            
            for (i,neuron1) in enumerate(self.source.pop.all()):
                pnv_source.append(self.source.get_neuron_annotation(i,'LGNAfferentOrientation'))
            
            
            if self.parameters.probabilistic:
                SpecificProbabilisticArborization(network, self.source, self.target, weights,self.parameters.specific_arborization,self.name).connect()
            else:
                SpecificArborization(network, self.source, self.target, weights,self.parameters.specific_arborization,self.name).connect()
            



def gabor(x1,y1,x2,y2,orientation,frequency,phase,size,aspect_ratio):
        from numpy import cos,sin
        X = (x1-x2)*cos(orientation) + (y1-y2)*sin(orientation)
        Y = -(x1-x2)*sin(orientation) + (y1-y2)*cos(orientation)
        ker = - (X*X + Y*Y*(aspect_ratio**2)) / (2*(size**2))
        return numpy.exp(ker)*numpy.cos(2*numpy.pi*X*frequency+phase)

class GaborConnector(MozaikComponent):
      """
      Connector that creates gabor projection. The individual gabor parameters are drawn from 
      distributions specified in the parameter set:
      
      `aspect_ratio`  -  aspect ratio of the gabor
      `size`          -  the size of the gabor  RFs in degrees of visual field
      `orientation`   -  the orientation of the gabor RFs
      `phase`         -  the phase of the gabor RFs
      `frequency`     -  the frequency of the gabor in degrees of visual field 
      
      Other parameters:
      
      `topological`          -  should the receptive field centers vary with the position of the given neurons in the target sheet
                                (note positions of neurons are always stored in visual field coordinates)
      `probabilistic`        -  should the weights be probabilistic or directly proportianal to the gabor profile
      `delay`                -  #ms/μm the delay on the projections 

      `or_map`             - is a orientation map supplied?
      `or_map_location`    - if or_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array
      `phase_map`          - is a phase map supplied?
      `phase_map_location` - if phase_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array
      """
      
      required_parameters = ParameterSet({
          'aspect_ratio': ParameterDist, #aspect ratio of the gabor
          'size':         ParameterDist, #the size of the gabor  RFs in degrees of visual field
          'orientation_preference':  ParameterDist, #the orientation preference of the gabor RFs (note this is the orientation of the gabor + pi/2)
          'phase':        ParameterDist, #the phase of the gabor RFs
          'frequency':    ParameterDist, #the frequency of the gabor in degrees of visual field 
          
          'topological': bool, # should the receptive field centers vary with the position of the given neurons 
                               # (note positions of neurons are always stored in visual field coordinates)
          'probabilistic': bool, # should the weights be probabilistic or directly proportianal to the gabor profile
          'delay': float,    #ms/μm the delay on the projections
          
          'specific_arborization' : ParameterSet,
          
          'or_map' : bool, # is a orientation map supplied?
          'or_map_location' : str, # if or_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array
          
          'phase_map' : bool, # is a phase map supplied?
          'phase_map_location' : str, # if phase_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array
          
      })
	    

      def __init__(self, network, lgn_on, lgn_off, target, parameters,name):
    	     MozaikComponent.__init__(self, network,parameters)
             import pickle
             self.name = name
             on = lgn_on.pop
             off = lgn_off.pop
             
             on_weights=[] 
             off_weights=[]
             
             t_size = target.size_in_degrees()
             or_map = None
             if self.parameters.or_map:
                f = open(self.parameters.or_map_location,'r')
                or_map = pickle.load(f)*numpy.pi
                coords_x = numpy.linspace(-t_size[0]/2.0,t_size[0]/2.0,numpy.shape(or_map)[0])    
                coords_y = numpy.linspace(-t_size[1]/2.0,t_size[1]/2.0,numpy.shape(or_map)[1])
                X,Y =  numpy.meshgrid(coords_x, coords_y)    
                or_map = NearestNDInterpolator(zip(X.flatten(),Y.flatten()), or_map.flatten())
                
             phase_map = None
             if self.parameters.phase_map:
                f = open(self.parameters.phase_map_location,'r')
                phase_map = pickle.load(f)   
                coords_x = numpy.linspace(-t_size[0]/2.0,t_size[0]/2.0,numpy.shape(phase_map)[0])    
                coords_y = numpy.linspace(-t_size[1]/2.0,t_size[1]/2.0,numpy.shape(phase_map)[1])    
                X,Y =  numpy.meshgrid(coords_x, coords_y)    
                phase_map = NearestNDInterpolator(zip(X.flatten(),Y.flatten()), phase_map.flatten())
             
             
             for (j,neuron2) in enumerate(target.pop.all()):
                 
                if or_map:
                   orientation = or_map(target.pop.positions[0][j],target.pop.positions[1][j])
                else: 
                   orientation = parameters.orientation_preference.next()[0]

                if phase_map:
                   phase = phase_map(target.pop.positions[0][j],target.pop.positions[1][j])
                else: 
                   phase = parameters.phase.next()[0] 

                # HACK!!!
                #if j == 0: 
                #   orientation = 0
                #   if target.name=='V1_Exc':
                #      phase = 0
                #   elif target.name=='V1_Inh':
                #      phase = 0
                
                aspect_ratio = parameters.aspect_ratio.next()[0]
                frequency = parameters.frequency.next()[0]
                size = parameters.size.next()[0]
				
                if orientation > numpy.pi:
                   print orientation
                
                target.add_neuron_annotation(j,'LGNAfferentOrientation',orientation,protected=True)
                target.add_neuron_annotation(j,'LGNAfferentAspectRatio',aspect_ratio,protected=True)
                target.add_neuron_annotation(j,'LGNAfferentFrequency',frequency,protected=True)
                target.add_neuron_annotation(j,'LGNAfferentSize',size,protected=True)
                target.add_neuron_annotation(j,'LGNAfferentPhase',phase,protected=True)
                 
                for (i,neuron1) in enumerate(on.all()):
                    if parameters.topological:
                        on_weights.append((i,j,numpy.max((0,gabor(on.positions[0][i],on.positions[1][i],target.pop.positions[0][j],target.pop.positions[1][j],orientation+numpy.pi/2,frequency,phase,size,aspect_ratio))),parameters.delay))
                        off_weights.append((i,j,-numpy.min((0,gabor(off.positions[0][i],off.positions[1][i],target.pop.positions[0][j],target.pop.positions[1][j],orientation+numpy.pi/2,frequency,phase,size,aspect_ratio))),parameters.delay))
                    else:
                        on_weights.append((i,j,numpy.max((0,gabor(on.positions[0][i],on.positions[1][i],0,0,orientation+numpy.pi/2,frequency,phase,size,aspect_ratio))),parameters.delay))
                        off_weights.append((i,j,-numpy.min((0,gabor(off.positions[0][i],off.positions[1][i],0,0,orientation+numpy.pi/2,frequency,phase,size,aspect_ratio))),parameters.delay))
             
             
             if parameters.probabilistic:
                 on_proj =  SpecificProbabilisticArborization(network,lgn_on,target,on_weights,parameters.specific_arborization,'ON_to_[' + target.name + ']')
                 off_proj = SpecificProbabilisticArborization(network,lgn_off,target,off_weights,parameters.specific_arborization,'OFF_to_[' + target.name + ']')
             else:
                 on_proj =  SpecificArborization(network,lgn_on,target,on_weights,parameters.specific_arborization,'ON_to_[' + target.name + ']')
                 off_proj = SpecificArborization(network,lgn_off,target,off_weights,parameters.specific_arborization,'OFF_to_[' + target.name + ']')
             
             on_proj.connect()
             off_proj.connect()
             #on_proj.connection_field_plot_continuous(0)   
             #off_proj.connection_field_plot_continuous(0)   

