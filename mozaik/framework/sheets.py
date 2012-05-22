# encoding: utf-8
"""
Sheet is an abstraction of a 2D continuouse sheet of neurons, roughly corresponding to the PyNN Population class with
the added spatial structure.
"""
import os
import numpy
from mozaik.framework.interfaces import MozaikComponent
import logging
import pyNN
import pyNN.recording.files

from NeuroTools.parameters import ParameterSet
from pyNN import random, space
from pyNN.errors import NothingToWriteError

from neo.core.spiketrain import SpikeTrain
from neo.core.segment import Segment
from neo.core.analogsignalarray import AnalogSignalArray

logger = logging.getLogger("mozaik")

class Sheet(MozaikComponent):
    """
    """
    
    required_parameters = ParameterSet({
        'cell': ParameterSet({
            'model': str, # the cell type of the sheet
            'params': ParameterSet,
            'initial_values': ParameterSet,
        }),
        
        'background_noise' : ParameterSet({
            # the background noise to the population. This will be generated as Poisson
            # note that this is optimized for NEST !!! 
            # it used native_cell_type("poisson_generator") to generate the noise
            
            'exc_firing_rate' : float, 
            'exc_weight' : float,
            'inh_firing_rate' : float,
            'inh_weight' : float,
        }),
        'name':str,
    })

    def __init__(self, model, parameters):
        """
        Sheet is an abstraction of a 2D continuouse sheet of neurons, roughly corresponding to the PyNN Population class with
        the added spatial structure.

        The spatial position of all cells is kept within the PyNN Population object. Each sheet is assumed to be centered around (0,0) origin,
        corresponding to whatever excentricity the model is looking at. The internal representation of space is degrees of visual field.
        Thus x,y coordinates of a cell in all sheets correspond to the degrees of visual field this cell is away from the origin.
        However, the sheet and derived classes/methods are supposed to accept parameters in units that are most natural for the given
        parameter and recalculate these into the internal degrees of visual field representation.

        The units in visual space should be in degrees.
        The units for cortical space should be in μm.
        The units for time are in ms.
        """
        MozaikComponent.__init__(self, model, parameters)
        self.sim = self.model.sim
        self.name = parameters.name # the name of the population
        self.model.register_sheet(self)
        self.to_record = None
        self._pop = None
    
    def size_in_degrees(self):
        """Returns the x,y size in degrees of visual field of the given area"""
        raise NotImplementedError
        pass
        
    def pop():
        doc = "PyNN population"

        def fget(self):
            if not self._pop:
                    print 'Population have not been yet set in sheet: ', self.name , '!'
            return self._pop
            
        def fset(self, value):
                if self._pop:
                   raise Exception("Error population has already been set. It is not allowed to do this twice!")
                self._pop = value
                self._neuron_annotations = [{} for i in xrange(0,len(value))]
                self.setup_background_noise()
        return locals()  
    
    pop = property(**pop()) #this will be populated by PyNN population, in the derived classes		
    
    def add_neuron_annotation(self,neuron_number,key,value,protected=True):
        if not self._pop:
            print 'Population have not been yet set in sheet: ', self.name , '!'
        if protected and self._neuron_annotations[neuron_number].has_key(key) and self._neuron_annotations[neuron_number][key][0]:
            print 'The annotation<', key , '> for neuron ' , str(i), ' is protected. Annotation not updated'
        else:
            self._neuron_annotations[neuron_number][key] =  (protected,value)
        
    def get_neuron_annotation(self,neuron_number,key):
        if not self._pop:
               print 'Population have not been yet set in sheet: ', self.name , '!'
        return self._neuron_annotations[neuron_number][key][1]
    
    def get_neuron_annotations(self):
        if not self._pop:
               print 'Population have not been yet set in sheet: ', self.name , '!'

        anns = []
        for i in xrange(0,len(self.pop)):
            d = {}
            for (k,v) in self._neuron_annotations[i].items():
                d[k] = v[1]
            anns.append(d)
        return anns
    
    def describe(self, template='default', render=lambda t,c: Template(t).safe_substitute(c)):
        context = {
            'name': self.__class__.__name__,

        }
        if template:
            render(template, context)
        else:
            return context

    def record(self):
        if self.to_record != None:
            for variable in self.to_record.keys():
                cells = self.to_record[variable]
                if cells != 'all':
                   self.pop[cells].record(variable)
                else:
                   self.pop.record(variable)
    
    def write_neo_object(self,stimulus_duration=None): 
        """
        Retrieve recorded data from pyNN. 
        In case offset is set it means we want to keep only data after time offset.
        """

        try:
            block = self.pop.get_data(['spikes','v','gsyn_exc','gsyn_inh'],clear=True)
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)
        s = block.segments[-1]   
        s.annotations["sheet_name"] = self.name
        
        # lets sort spike train so that it is ordered by IDs and thus hopefully population indexes
        def compare(a, b):
            return cmp(a.annotations['source_id'], b.annotations['source_id']) 
        

        
        s.spiketrains = sorted(s.spiketrains, compare)
        if stimulus_duration != None:
           end = s.spiketrains[0].t_stop
           start = s.spiketrains[0].t_stop - stimulus_duration * s.spiketrains[0].t_stop.units
           for i in xrange(0,len(s.spiketrains)): 
               sp = s.spiketrains[i].time_slice(start,end).copy() - start
               s.spiketrains[i] = SpikeTrain(sp.magnitude * start.units,t_start = 0 * start.units, t_stop = stimulus_duration * start.units,name = s.spiketrains[i].name,description = s.spiketrains[i].description, file_origin = s.spiketrains[i].file_origin, **s.spiketrains[i].annotations)

           for i in xrange(0,len(s.analogsignalarrays)):
               s.analogsignalarrays[i] = s.analogsignalarrays[i].time_slice(start,end).copy()
               s.analogsignalarrays[i].t_start = 0 * start.units
        
        return s
    
    def setup_background_noise(self):
        from pyNN.nest import native_cell_type
        if (self.parameters.background_noise.exc_firing_rate != 0) or (self.parameters.background_noise.exc_firing_rate != 0):
            np_exc = self.sim.Population(1, native_cell_type("poisson_generator"), {'rate' : self.parameters.background_noise.exc_firing_rate})
            prj = self.sim.Projection(np_exc, self.pop, self.sim.AllToAllConnector(weights=self.parameters.background_noise.exc_weight),target='excitatory')

        if (self.parameters.background_noise.inh_firing_rate != 0) or (self.parameters.background_noise.inh_firing_rate != 0):
            np_inh = self.sim.Population(1, native_cell_type("poisson_generator"), {'rate' : self.parameters.background_noise.inh_firing_rate})
            prj = self.sim.Projection(np_inh, self.pop, self.sim.AllToAllConnector(weights=self.parameters.background_noise.inh_weight),target='inhibitory')

        
class RetinalUniformSheet(Sheet):

    required_parameters = ParameterSet({
        'sx': float, #degrees, x size of the region
        'sy': float, #degrees, y size of the region
        'density': float, # neurons/(degree^2)
    })

    def __init__(self, model, parameters):
        """
        """
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__,int(parameters.sx*parameters.sy*parameters.density)))
        Sheet.__init__(self, model, parameters)
        rs = space.RandomStructure(boundary=space.Cuboid(parameters.sx,parameters.sy,0),origin=(0.0, 0.0, 0.0))
        self.pop = self.sim.Population(int(parameters.sx*parameters.sy*parameters.density), getattr(self.model.sim, self.parameters.cell.model), self.parameters.cell.params,rs,label=self.name)
        for var, val in self.parameters.cell.initial_values.items():
            self.pop.initialize(var, val)
            
    def size_in_degrees(self):            
        return (self.parameters.sx, self.parameters.sy)


class SheetWithMagnificationFactor(Sheet):

    required_parameters = ParameterSet({
        'magnification_factor': float, # μm / degree
        'sx': float,      #μm, x size of the region
        'sy': float,      #μm, y size of the region
    })

    def __init__(self, model, parameters):
        """
        """
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__,int(parameters.sx*parameters.sy/10000*parameters.density)))
        Sheet.__init__(self, model, parameters)
        self.magnification_factor = parameters.magnification_factor

    def vf_2_cs(self,degree_x,degree_y):
        """
        vf_2_cs converts the position (degree_x,degree_y) in visual field to position in cortical space (in μm)
        given the magnification_factor.
        """
        return (degree_x*self.magnification_factor,degree_y*self.magnification_factor)

    def cs_2_vf(self,micro_meters_x,micro_meters_y):
        """
        cs_2_vf converts the position (micro_meters_x,micro_meters_y) in cortical space to the position in the
        visual field (in degrees) given the magnification_factor
        """
        return (micro_meters_x/self.magnification_factor,micro_meters_x/self.magnification_factor)

    def dvf_2_dcs(self,distance_vf):
        """
        dvf_2_dcs converts the distance in visual space to the distance in cortical space given the magnification_factor
        """
        return distance_vf*self.magnification_factor

    def size_in_degrees(self):            
        return self.cs_2_vf(self.parameters.sx, self.parameters.sy)


class CorticalUniformSheet(SheetWithMagnificationFactor):

    required_parameters = ParameterSet({
        'density': float, #neurons/(100 μm^2)
    })

    def __init__(self, model, parameters):
        """
        """
        SheetWithMagnificationFactor.__init__(self, model, parameters)
        dx,dy = self.cs_2_vf(parameters.sx,parameters.sy)
        
        rs = space.RandomStructure(boundary=space.Cuboid(dx,dy,0),origin=(0.0, 0.0, 0.0))
        self.pop = self.sim.Population(int(parameters.sx*parameters.sy/10000*parameters.density), getattr(self.model.sim, self.parameters.cell.model), self.parameters.cell.params,rs,label=self.name)
        
        for var, val in self.parameters.cell.initial_values.items():
            self.pop.initialize(var, val)

