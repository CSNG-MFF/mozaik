# encoding: utf-8
"""
Sheet is an abstraction of a 2D continuouse sheet of neurons, roughly corresponding to the PyNN Population class with
the added spatial structure.
"""
import os
import numpy
from MozaikLite.framework.interfaces import MozaikComponent
import logging
import pyNN
import pyNN.recording.files
import quantities
from NeuroTools import visual_logging
from NeuroTools.parameters import ParameterSet
from pyNN import random, space
from pyNN.errors import NothingToWriteError

from neo.core.spiketrain import SpikeTrain
from neo.core.segment import Segment
from neo.core.analogsignal import AnalogSignal
from MozaikLite.tools.misc import get_spikes_to_dic, get_vm_to_dic,get_gsyn_to_dicts

logger = logging.getLogger("Mozaik")

class Sheet(MozaikComponent):
    """
    """

    required_parameters = ParameterSet({
        'cell': ParameterSet({
            'model': str, # the cell type of the sheet
            'params': ParameterSet,
            'initial_values': ParameterSet,
        }),

        'name':str,
    })

    pop = None # this will be populated by PyNN population, in the derived classes

    def __init__(self, network, parameters):
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
        MozaikComponent.__init__(self, network, parameters)
        self.network.register_sheet(self)
        self.sim = self.network.sim
        self.name = parameters.name # the name of the population
        self.to_record = False
    
    def describe(self, template='default', render=lambda t,c: Template(t).safe_substitute(c)):
        context = {
            'name': self.__class__.__name__,

        }
        if template:
            render(template, context)
        else:
            return context

    def record(self, variable, cells='all'):
        func_mapping = {'spikes': 'record', 'v': 'record_v','g_syn':'record_gsyn'} # need to add conductances
        record_method = func_mapping[variable]
        if cells == 'all':
            logger.debug('Recording %s from all cells in population "%s"' % (variable, self.name))
            getattr(self.pop, record_method)()
        elif isinstance(cells, dict):
            logger.debug('Recording %s from a subset of cells in population "%s" ' % (variable, self.name))
            getattr(self.pop, record_method)(cells[name])
        elif isinstance(cells, int):
            n = cells
            logger.debug('Recording %s from a subset of %d cells in population "%s" ' % (variable, n, self.name))
            getattr(self.pop, record_method)(n)
        else:
            raise Exception("cells must be 'all', a dict, or an int. Actual value of %s" % str(cells))

    def write_neo_object(self,segment,tstop):
        try:
            spikes = get_spikes_to_dic(self.pop.getSpikes(),self.pop)
            for k in spikes.keys():
                # it assumes segment implements and add function which takes the id of a neuron and the corresponding its SpikeTrain
                st = SpikeTrain(spikes[k],t_start=0,t_stop=tstop,units=quantities.ms)
                st.index = k
                segment.spiketrains.append(st)
            logging.debug("Writing spikes from population %s to neo object." % (self.pop))
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)
        try:
            v = get_vm_to_dic(self.pop.get_v(),self.pop)
            for k in v.keys():
                # it assumes segment implements and add function which takes the id of a neuorn and the corresponding its SpikeTrain
                st = AnalogSignal(v[k],units=quantities.mV,sampling_period=self.network.sim.get_time_step()*quantities.ms)
                st.index = k
                segment.analogsignals.append(st)
                segment.annotations['vm'].append(len(segment.analogsignals)-1)
            logging.debug("Writing Vm from population %s to neo object." % (self.pop))
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)
        try:
            gsyn_e,gsyn_i = get_gsyn_to_dicts(self.pop.get_gsyn(),self.pop)
            for k in v.keys():
                # it assumes segment implements and add function which takes the id of a neuorn and the corresponding its SpikeTrain
                st_e = AnalogSignal(0.000001*gsyn_e[k],sampling_period=self.network.sim.get_time_step()*quantities.ms,units=quantities.S)
                st_i = AnalogSignal(0.000001*gsyn_i[k],sampling_period=self.network.sim.get_time_step()*quantities.ms,units=quantities.S)
                st_e.index = k
                st_i.index = k
                segment.analogsignals.append(st_e)
                segment.annotations['gsyn_e'].append(len(segment.analogsignals)-1)
                segment.analogsignals.append(st_i)
                segment.annotations['gsyn_i'].append(len(segment.analogsignals)-1)
            logging.debug("Writing Vm from population %s to neo object." % (self.pop))
        except NothingToWriteError, errmsg:
            logger.debug(errmsg)

        
        return segment

class RetinalUniformSheet(Sheet):

    required_parameters = ParameterSet({
        'sx': float, #degrees, x size of the region
        'sy': float, #degrees, y size of the region
        'density': float, # neurons/(degree^2)
    })

    def __init__(self, network, parameters):
        """
        """
        logger.info("Creating %s" % self.__class__.__name__)
        Sheet.__init__(self, network, parameters)
        rs = space.RandomStructure(boundary=space.Cuboid(parameters.sx,parameters.sy,0),origin=(0.0, 0.0, 0.0))
        self.pop = self.sim.Population(self, parameters.sx*parameters.sy*density, getattr(sim, self.parameters.cell.model), self.parameters.cell.model.params,rs,self.name)
        for var, val in self.parameters.cell.initial_values.items():
            self.pop.initialize(var, val)


class SheetWithMagnificationFactor(Sheet):

    required_parameters = ParameterSet({
        'magnification_factor': float, # μm / degree
    })

    def __init__(self, network, parameters):
        """
        """
        logger.info("Creating %s with %d neurons." % (self.__class__.__name__,int(parameters.sx*parameters.sy/10000*parameters.density)))
        Sheet.__init__(self, network, parameters)
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


class CorticalUniformSheet(SheetWithMagnificationFactor):

    required_parameters = ParameterSet({
        'sx': float,      #μm, x size of the region
        'sy': float,      #μm, y size of the region
        'density': float, #neurons/(100 μm^2)
    })

    def __init__(self, network, parameters):
        """
        """
        SheetWithMagnificationFactor.__init__(self, network, parameters)
        dx,dy = self.cs_2_vf(parameters.sx,parameters.sy)
        rs = space.RandomStructure(boundary=space.Cuboid(dx,dy,0),origin=(0.0, 0.0, 0.0))
        self.pop = self.sim.Population(int(parameters.sx*parameters.sy/10000*parameters.density), getattr(self.network.sim, self.parameters.cell.model), self.parameters.cell.params,rs,self.name)
        
        for var, val in self.parameters.cell.initial_values.items():
            self.pop.initialize(var, val)

