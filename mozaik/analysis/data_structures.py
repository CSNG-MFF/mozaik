"""
This module contains the definition of the AnalysisDataStructure API and implementation of some basic analysis data structures.

For more documentation refer to `mozaik.analysis`_
"""

import mozaik
import numpy
from mozaik.tools.mozaik_parametrized import MozaikParametrized, SNumber, SInteger, SString
logger = mozaik.getMozaikLogger()



class AnalysisDataStructure(MozaikParametrized):
    """
    Encapsulates data that a certain Analysis class generates.

    The four parameters that are common to all AnalysisDataStructure classes are `identified`, `analysis_algorithm`, `neuron`, `sheet_name` and `stimulus_id`.
    """

    identifier = SString(doc="The identifier of the analysis data structure")
    analysis_algorithm = SString(doc="The identifier of the analysis data structure")
    
    neuron = SInteger(default=None,
                           doc="Neuron id to which the datastructure belongs. None if it is not neuron specific")
    sheet_name = SString(default=None,
                              doc="The sheet for which this results were computed. None if they do not belong to specific sheet")
    stimulus_id = SString(default=None,
                               doc="The stimulus for which the results were computed. None if they are not related to specific stimulus")

    def __init__(self,tags=[], **params):
        MozaikParametrized.__init__(self, **params)
        self.tags = tags



class SingleValue(AnalysisDataStructure):
    """
    Data structure holding single value. This can be per model, if sheet parameter is None,
    or per sheet if sheet is specified. In principle it can also be per neuron if the neuron
    parameter is specified, but in most cases you probably want to use :class:`.PerNeuronValue`
    instead.
    """

    value = SNumber(units=None,default=None,doc="The value.")
    value_name = SString(doc="The name of the value.")
    period = SNumber(units=None,default=None,doc="The period of the value. If value is not periodic period=None")
    
    def __init__(self, **params):
        AnalysisDataStructure.__init__(self, identifier='SingleValue', **params)



class PerNeuronValue(AnalysisDataStructure):
    """
    Data structure holding single value per neuron.
    
    Parameters
    ---------- 
    
    values : list
           The vector of values one per neuron

    value_units : quantities
                Quantities unit describing the units of the value
    
    ids : list(int)
        The ids of the neurons which are stored, in the same order as in the values
    """
    value_name = SString(doc="The name of the value.")
    period = SNumber(units=None,default=None,doc="The period of the value. If value is not periodic period=None")

    def __init__(self, values, idds, value_units, **params):
        AnalysisDataStructure.__init__(self, identifier='PerNeuronValue', **params)
        self.value_units = value_units
        self.values = numpy.array(values)
        self.ids = list(idds)
        assert len(values) == len(idds), '%s %s' % (str(values),str(idds))
    
    def get_value_by_id(self,idds):
        """
        Parameters
        ---------- 
        idd : int or list(int)
            The ids for which the return the values.
        
        Returns
        -------
        ids : AnalogSignal or list(AnalogSignal)
            List (or single) of AnalogSignal objects corresponding to ids in `idd`.
        """
        if isinstance(idds,list) or isinstance(idds,numpy.ndarray):
            return [self.values[list(self.ids).index(i)] for i in idds]
        else:
            return numpy.array(self.values)[list(self.ids).index(idds)]



class PerNeuronPairValue(AnalysisDataStructure):
    """
    Data structure holding values for each pair of neurons.
    
    Parameters
    ---------- 
    
    values : numpy.nd_array
           The 2D array holding the values. 

    value_units : quantities
                Quantities unit describing the units of the value
    
    ids : list(int)
        The ids of the neurons which are stored, in the same order as in the values (along both axis).
    """
    value_name = SString(doc="The name of the value.")
    period = SNumber(units=None,default=None,doc="The period of the value. If value is not periodic period=None")

    def __init__(self, values, idds, value_units, **params):
        AnalysisDataStructure.__init__(self, identifier='PerNeuronValue', **params)
        self.value_units = value_units
        self.values = numpy.array(values)
        self.ids = list(idds)
        assert values.shape == (len(idds),len(idds))
    
    def get_value_by_ids(self,idds1,idds2):
        """
        Parameters
        ---------- 
        idds1 : int or list(int)
            The ids for which the return the values along first dimension.
        
        idds2 : int or list(int)
            The ids for which the return the values along first dimension.
        
        Returns
        -------
        ids : scaler or array
            Array or scalar of values corresponding to `idds`.
        """
        if (isinstance(idds1,list) or isinstance(idds1,numpy.ndarray)) and (isinstance(idds2,list) or isinstance(idds2,numpy.ndarray)):
            return numpy.array(self.values)[[list(self.ids).index(i) for i in idds1],:][:,[list(self.ids).index(i) for i in idds2]]
        else:
            return self.values[list(self.ids).index(idds1),list(self.ids).index(idds2)]



class AnalysisDataStructure1D(AnalysisDataStructure):
    """
    Data structure representing 1D data.
    All data corresponds to the same axis name and units.
    Explicitly specifies the axis - their name and units.
    Note that at this stage we do not assume the structure in which the data
    is stored.
    Also the class can hold multiple instances of 1D data.

    It uses the quantities package to specify units.
    If at all possible all data stored in numpy arrays should be quantities
    arrays with matching units!

    Parameters
    ---------- 
    y_axis_units : quantities
          The quantities units of y axis.
    """

    x_axis_name = SString(doc="the name of the x axis.")
    y_axis_name = SString(doc="the name of the y axis.")

    def __init__(self,x_axis_units,y_axis_units,**params):
        AnalysisDataStructure.__init__(self, **params)
        self.y_axis_units = y_axis_units
        self.x_axis_units = x_axis_units



class AnalogSignal(AnalysisDataStructure1D):
    """
    A single analog signal. This is effectively a wrapper around a single Neo AnalogSignal object.

    Parameters
    ---------- 
    analog_signal : AnalogSignal
         The neo AnalogSignal
    """

    def __init__(self, analog_signal, y_axis_units, **params):
        AnalysisDataStructure1D.__init__(self,  analog_signal.sampling_period.units,y_axis_units,
                                         identifier='AnalogSignal',
                                         **params)
        self.analog_signal = analog_signal
    
    def __add__(self, other):
        assert self.x_axis_name == other.x_axis_name
        assert self.y_axis_name == other.y_axis_name
        assert self.y_axis_units == other.y_axis_units
        return AnalogSignal(self.analog_signal+other.analog_signal,y_axis_units = self.y_axis_units,x_axis_name = self.x_axis_name,y_axis_name = self.y_axis_name, sheet_name = self.sheet_name)



class AnalogSignalList(AnalysisDataStructure1D):
    """
    This is a simple list of Neo AnalogSignal objects.

    Parameters
    ---------- 
    asl : list(AnalogSignal)
         The variable containing the list of AnalogSignal objects, in the order
         corresponding to the order of neurons indexes in the indexes parameter.
    
    ids : list(int)
         List of ids of neurons in the original Mozaik sheet to which the
         AnalogSignals correspond.
    """

    def __init__(self, asl, ids, y_axis_units, **params):
        AnalysisDataStructure1D.__init__(self,  asl[0].sampling_period.units,y_axis_units,
                                         identifier='AnalogSignalList',
                                         **params)
        self.asl = asl
        self.ids = list(ids)
        assert len(asl) == len(ids)
    
    def get_asl_by_id(self,idds):
        """
        Parameters
        ---------- 
        idd : int or list(int)
        
        Returns
        -------
        asls : AnalogSignal or list(AnalogSignal)
            List (or single) of AnalogSignal objects corresponding to ids in `idds`.
        """
        if isinstance(idds,list) or isinstance(idds,numpy.ndarray):
            return [self.asl[list(self.ids).index(i)] for i in idds]
        else:
            return self.asl[list(self.ids).index(idds)]
    
    def __add__(self, other):
        assert set(self.ids) <= set(other.ids) and set(self.ids) >= set(other.ids)  
        assert self.x_axis_name == other.x_axis_name
        assert self.y_axis_name == other.y_axis_name
        assert self.y_axis_units == other.y_axis_units
        
        new_asl = []
        for idd in self.ids:
            new_asl.append(self.get_asl_by_id(idd) + other.get_asl_by_id(idd))
            
        return AnalogSignalList(new_asl,self.ids,y_axis_units = self.y_axis_units,x_axis_name = self.x_axis_name,y_axis_name = self.y_axis_name, sheet_name = self.sheet_name)
    
    def mean(self):
        """
        Calculates the mean analog signal from the ones in the list.
        """
        for asl in self.asl:
            assert asl.units == self.asl[0].units, "AnalogSignalList.mean: units of AnalogSignal objects in the list do not match."
            assert asl.sampling_rate == self.asl[0].sampling_rate, "AnalogSignalList.mean: sampling_rate of AnalogSignal objects in the list do not match"
            assert asl.t_start == self.asl[0].t_start, "AnalogSignalList.mean: t_start of AnalogSignal objects in the list do not match."        
        
        return numpy.mean(self.asl,axis=0)

    def var(self):
        """
        Calculates the mean analog signal from the ones in the list.
        """
        for asl in self.asl:
            assert asl.units == self.asl[0].units, "AnalogSignalList.mean: units of AnalogSignal objects in the list do not match."
            assert asl.sampling_rate == self.asl[0].sampling_rate, "AnalogSignalList.mean: sampling_rate of AnalogSignal objects in the list do not match"
            assert asl.t_start == self.asl[0].t_start, "AnalogSignalList.mean: t_start of AnalogSignal objects in the list do not match."        
        
        return numpy.var(self.asl,axis=0)



class PerNeuronPairAnalogSignalList(AnalysisDataStructure1D):
    """
    This is a list of Neo AnalogSignal objects associated with pairs of neurons.

    Parameters
    ---------- 
    asl : list(AnalogSignal)
         The variable containing the list of AnalogSignal objects, in the order
         corresponding to the order of neuron indexe pairs in the ids parameter.
    
    ids : list((int,int))
         List of id pairs of neurons to which the AnalogSignals correspond.
    """

    def __init__(self, asl, ids, y_axis_units, **params):
        AnalysisDataStructure1D.__init__(
            self,  
            asl[0].sampling_period.units, 
            y_axis_units,
            identifier='AnalogSignalList',
            **params
        )
        self.asl = asl
        self.ids = list(ids)
        assert len(asl) == len(ids)
    
    def get_asl_by_id_pair(self,idd_pair):
        """
        Parameters
        ---------- 
        idd_pair : tuple of neuron ids or list of tuples of neuron ids
        
        Returns
        -------
        ids : AnalogSignal or list(AnalogSignal)
            List (or single) of AnalogSignal objects corresponding to id pairs specified in `idd_pair` parameter.
        """
        return self.asl[list(self.ids).index(idd_pair)]

    def __add__(self, other):
        assert set(self.ids) <= set(other.ids) and set(self.ids) >= set(other.ids)  
        assert self.x_axis_name == other.x_axis_name
        assert self.y_axis_name == other.y_axis_name
        assert self.y_axis_units == other.y_axis_units
        
        new_asl = []
        for idd in self.ids:
            new_asl.append(self.get_asl_by_id_pair(idd) + other.get_asl_by_id_pair(idd))
            
        return PerNeuronPairAnalogSignalList( new_asl, self.ids, analysis_algorithm=self.analysis_algorithm, y_axis_units=self.y_axis_units, x_axis_name=self.x_axis_name, y_axis_name=self.y_axis_name, sheet_name=self.sheet_name, stimulus_id=self.stimulus_id )

    def __sub__(self, other):
        assert set(self.ids) <= set(other.ids) and set(self.ids) >= set(other.ids)  
        assert self.x_axis_name == other.x_axis_name
        assert self.y_axis_name == other.y_axis_name
        assert self.y_axis_units == other.y_axis_units
        
        new_asl = []
        for idd in self.ids:
            new_asl.append(self.get_asl_by_id_pair(idd) - other.get_asl_by_id_pair(idd))
            
        return PerNeuronPairAnalogSignalList( new_asl, self.ids, analysis_algorithm=self.analysis_algorithm, y_axis_units=self.y_axis_units, x_axis_name=self.x_axis_name, y_axis_name=self.y_axis_name, sheet_name=self.sheet_name, stimulus_id=self.stimulus_id )

    def division_by_num(self, num):
        """
        Divides all asl by the supplied number.
        """
        for asl in self.asl:
            assert asl.units == self.asl[0].units, "AnalogSignalList.mean: units of AnalogSignal objects in the list do not match."
            assert asl.sampling_rate == self.asl[0].sampling_rate, "AnalogSignalList.mean: sampling_rate of AnalogSignal objects in the list do not match"
            assert asl.t_start == self.asl[0].t_start, "AnalogSignalList.mean: t_start of AnalogSignal objects in the list do not match."        
        
        assert num!=0.0

        new_asl = []
        for idd in self.ids:
            new_asl.append(self.get_asl_by_id_pair(idd) / num)

        return PerNeuronPairAnalogSignalList( new_asl, self.ids, analysis_algorithm=self.analysis_algorithm, y_axis_units=self.y_axis_units, x_axis_name=self.x_axis_name, y_axis_name=self.y_axis_name, sheet_name=self.sheet_name, stimulus_id=self.stimulus_id )

    def mean(self):
        """
        Calculates the mean analog signal from the ones in the list.
        """
        for asl in self.asl:
            assert asl.units == self.asl[0].units, "AnalogSignalList.mean: units of AnalogSignal objects in the list do not match."
            assert asl.sampling_rate == self.asl[0].sampling_rate, "AnalogSignalList.mean: sampling_rate of AnalogSignal objects in the list do not match"
            assert asl.t_start == self.asl[0].t_start, "AnalogSignalList.mean: t_start of AnalogSignal objects in the list do not match."        
        
        return numpy.sum(self.asl)/len(self.asl)

        
        
class ConductanceSignalList(AnalysisDataStructure1D):
    """
    This is a simple list of Neurotools AnalogSignal objects representing the
    conductances.

    The object holds two lists, one for excitatory and one for inhibitory
    conductances.

    Parameters
    ---------- 
    e_asl : list(AnalogSignal)
       The variable containing the list of AnalogSignal objects corresponding
       to excitatory conductances, in the order corresponding to the order of
       neurons indexes in the indexes parameter.
    
    i_asl : list(AnalogSignal)
       The variable containing the list of AnalogSignal objects corresponding
       to inhibitory conductances, in the order corresponding to the order of
       neurons indexes in the indexes parameter.
       
    ids : list(int)
       List of ids of neurons in the original Mozaik sheet to which the
       AnalogSignals correspond.
    """

    def __init__(self, e_con, i_con, ids, **params):
        assert e_con[0].units == i_con[0].units
        AnalysisDataStructure1D.__init__(self,
                                         e_con[0].sampling_period.units,
                                         e_con[0].units,
                                         x_axis_name='time',
                                         y_axis_name='conductance',
                                         identifier='ConductanceSignalList',
                                         **params)
        self.e_con = e_con
        self.i_con = i_con
        self.ids = list(ids)
        assert len(e_con) == len(ids)
        assert len(i_con) == len(ids)
        
    def get_econ_by_id(self,idd):
        """
        Parameters
        ---------- 
        idd : int or list(int)
        
        Returns
        -------
        ids : AnalogSignal or list(AnalogSignal)
            List (or single) of AnalogSignal objects containing excitatory conductanes corresponding to ids in `idd`.
        """
        return self.e_con[self.ids.index(idd)]

    def get_icon_by_id(self,idd):
        """
        Parameters
        ---------- 
        idd : int or list(int)
        
        Returns
        -------
        ids : AnalogSignal or list(AnalogSignal)
            List (or single) of AnalogSignal objects containing inhibitory conductanes corresponding to ids in `idd`.
        """
        return self.i_con[self.ids.index(idd)]

    def mean(self):
        """
        Calculates the mean conductance from the ones in the list.
        """
        for asl in self.e_con:
            assert asl.units == self.asl[0].units, "ConductanceSignalList.mean: units of AnalogSignal objects in the exc. list do not match."
            assert asl.sampling_rate == self.asl[0].sampling_rate, "ConductanceSignalList.mean: sampling_rate of AnalogSignal objects in the exc.list do not match"
            assert asl.t_start == self.asl[0].t_start, "ConductanceSignalList.mean: t_start of AnalogSignal objects in the exc. list do not match."        

        for asl in self.i_con:
            assert asl.units == self.asl[0].units, "ConductanceSignalList.mean: units of AnalogSignal objects in the inh. list do not match."
            assert asl.sampling_rate == self.asl[0].sampling_rate, "ConductanceSignalList.mean: sampling_rate of AnalogSignal objects in the inh. list do not match"
            assert asl.t_start == self.asl[0].t_start, "ConductanceSignalList.mean: t_start of AnalogSignal objects in the inh. list do not match."        

        return (numpy.sum(self.e_con)/len(self.e_con),numpy.sum(self.i_con)/len(self.i_con))


class Connections(AnalysisDataStructure):
    """
    Data structure holding connections.

    Parameters
    ---------- 
    weights : list
            List of tuples (i,j,w) where i is index of pre-synaptic neuron in sheet source_name and j is index of post-synaptic neuron in sheet target_name, and w is the weights.
    
    delays : list
            List of tuples (i,j,d) where i is index of pre-synaptic neuron in sheet source_name and j is index of post-synaptic neuron in sheet target_name, and d is the delay.
    
    """

    proj_name = SString(doc="Projection name.")
    source_name = SString(doc="The name of the source sheet.")
    target_name = SString(doc="The name of the target sheet.")

    def __init__(self, weights, delays, source_size, target_size, **params):
        AnalysisDataStructure.__init__(self, identifier='Connections', **params)
        self.weights = weights
        self.delays =  delays
        self.source_size = source_size
        self.target_size = target_size
