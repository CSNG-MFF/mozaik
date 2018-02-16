"""
This module implements the data storage functionality.
"""

import numpy
from parameters import ParameterSet
from neo.core.block import Block
#from neo.io.hdf5io import NeoHdf5IO
import mozaik
from mozaik.core import ParametrizedObject
from neo_neurotools_wrapper import MozaikSegment, PickledDataStoreNeoWrapper
from mozaik.tools.mozaik_parametrized import  MozaikParametrized,filter_query
import cPickle
import collections
import os.path

logger = mozaik.getMozaikLogger()

class DataStoreView(ParametrizedObject):
    """
    This class represents a subset of a DataStore and defines the query
    interface and the structure in which the data are stored in the memory of
    any datastore. Main role of this class is to allow for creating subsets of
    data stored in the :class:`.DataStore`, so that one can restrict other parts of
    Mozaik to work only over these subsets. This is done via means of queries
    (see :mod:`mozaik.storage.queries`) which produce :class:`.DataStoreView` objects and can be
    chained to work like filters.

    Note that the actual datastores inherit from this object and define how the
    data are actualy stored on other media (i.e. hdf5 file format or simple pickle).

    Data store should aggregate all data and analysis results collected across
    experiments and a given model.

    Experiments results storage:

    These are stored as a simple list in the self.block.segments variable. Each
    segment corresponds to recordings from a single sheet to a single stimulus.

    Analysis results storage:
    
    A list containing the :class:`.mozaik.analysis.data_structures.AnalysisDataStructure` objects.

    The analysis results are addressed by the
    AnalysisDataStructure identifier. The call with this specification returns
    a set of AnalysisDataStructures that correspond to the above addressing.
    If further more specific 'addressing' is required it
    has to be done by the corresponding visualization or analysis code that
    asked for the AnalysisDataStructure's based on the knowledge of their
    content. Or a specific query filters can be written that understand the specific type
    of AnalysisDataStructure and can filter them based on their internal data.
    For more details on addressing experimental results or analysis data structures
    please reffer to :mod:`.queries` or :py:mod:`mozaik.tools.mozaik_parametrized` modules.

    DataStoreView also keeps a reference to a full `.Datastore` object
    from which it was originally created (this might have happened via a
    chain of DSVs). This is on order to allow for operations that work over DSV
    to insert their results into the original full datastore as this is
    (almost?) always the desired behaviours (note DSV does not actually have
    functions to add new recordings or analysis results).
    
    By default, the datastore will refuse to add a new AnalysisDataStructure 
    to the datastore if the new ADS has the same values of all its parameters as
    some other ADS already inserted in the datastore. This is so that each ADS
    stored in datastore is uniquely identifiable based on its parameters.
    If the datastore is created (loaded) with the replace flag set to True, in the situation
    of such conflict the datastore will replace the new ADS for the one already in the datastore.
    """

    def __init__(self, parameters, full_datastore,replace=False):
        ParametrizedObject.__init__(self, parameters)
        # we will hold the recordings as one neo Block
        self.block = Block()
        self.analysis_results = []
        self.replace = replace
        self.sensory_stimulus = collections.OrderedDict()
        self.full_datastore = full_datastore  # should be self if actually the
                                              # instance is actually DataStore

    def get_segments(self,null=False):
        """
        Returns list of all recordings (as neo segments) stored in the datastore.
        """
        return [s for s in self.block.segments if s.null == null]
        
    def sheets(self):
        """
        Returns the list of all sheets that are present in at least one of the
        segments in the given DataStoreView.
        """
        sheets = collections.OrderedDict()
        for s in self.block.segments:
            sheets[s.annotations['sheet_name']] = 1
        
        for ads in self.analysis_results:
            sheets[ads.sheet_name] = 1
        
        if sheets.has_key(None):
            sheets.pop(None)
                
        return sheets.keys()

    def get_neuron_postions(self):
        """
        Returns the positions for all neurons in the model within their respective sheets.
        A dictionary is returned with keys names of sheets and values a 2d ndarray of size (2,number of neurons)
        holding the x and y positions of all neurons in the rows.
        
        Use :func:`.get_sheet_indexes` to link the indexes in the returned array to neuron idds.
        """
        return self.full_datastore.block.annotations['neuron_positions']

    def get_sheet_indexes(self, sheet_name,neuron_ids):
        """
        Returns the indexes of neurons in the sheet given the idds (this should be primarily used with annotations data such as positions etc.)
        """
        ids = self.full_datastore.block.annotations['neuron_ids'][sheet_name]
        if isinstance(neuron_ids,list) or isinstance(neuron_ids,numpy.ndarray):
          return [numpy.where(ids == i)[0][0] for i in neuron_ids]
        else:
          return numpy.where(ids == neuron_ids)[0][0]

    def get_sheet_ids(self, sheet_name,indexes=None):
        """
        Returns the idds of neurons in the sheet given the indexes (this should be primarily used with annotations data such as positions etc.)
        """
        # find first segment from sheet sheet_name
        if isinstance(indexes,list) or isinstance(indexes,numpy.ndarray):
            return self.full_datastore.block.annotations['neuron_ids'][sheet_name][indexes]
        elif indexes  == None:
            return self.full_datastore.block.annotations['neuron_ids'][sheet_name]
        else:
            raise ValueError("indexes can be aither None or list or ndarray, %s was supplied instead" % (str(type(indexes))))
    def get_sheet_parameters(self,sheet_name):
        """
        Returns the *ParemterSet* instance corresponding to the given sheet.
        """
        return eval(self.full_datastore.block.annotations['sheet_parameters'])[sheet_name]


    def get_model_parameters(self):
        """
        Returns the *ParemterSet* instance corresponding to the whole model.
        """
        return self.full_datastore.block.annotations['model_parameters']


    def get_neuron_annotations(self):
        """
        Returns neuron annotations.
        """
        return self.full_datastore.block.annotations['neuron_annotations']

    def get_stimuli(self,null=False):
        """
        Returns a list of stimuli (as strings). The order of the stimuli
        corresponds to the order of segments returned by the get_segments()
        call.
        
        If *null* is true the order corresponds to the order of segments 
        returned by get_segments(null=True).
        """
        return [s.annotations['stimulus'] for s in self.block.segments if s.null == null]

    def get_analysis_result(self, **kwargs):
        """
        Return a list of ADSs, that match the parameter values specified in kwargs.
        
        Examples
        --------
        >>> datastore.get_analysis_result(identifier=['PerNeuronValue','SingleValue'],sheet_name=sheet,value_name='orientation preference')
        
        This command should return or ADS whose identifier is *PerNeuronValue* or *SingleValue*, and are associated with sheet named *sheet* and as their value name have 'orientation preference'
        """
        return filter_query(self.analysis_results,**kwargs)

    def get_sensory_stimulus(self, stimuli=None):
        """
        Return the raw sensory stimulus that has been presented to the model due to stimuli specified by the stimuli argument.
        If stimuli==None returns all sensory stimuli.
        """
        if stimuli == None:
            return self.sensory_stimulus.values()
        else:
            return [self.sensory_stimulus[s] for s in stimuli]

    def get_experiment_parametrization_list(self):
        
        """
        Return the list of parameters of all experiments performed (in the order they were performed).
        
        The returned data are in the following format:  a list of tuples (experimenta_class,parameter_set) where
        *experiment_class* is the class of the experiment, and parameter_set is a ParameterSet instance converted to string
        that corresponds to the parameters of the given experiment.
        """
        return self.block.annotations['experiment_parameters'];

    def sensory_stimulus_copy(self):
        """
        Utility function that makes a shallow copy of the dictionary holding sensory stimuli.
        """
        new_dict = collections.OrderedDict()
        for k in self.sensory_stimulus.keys():
            new_dict[k] = self.sensory_stimulus[k]
        return new_dict

    def analysis_result_copy(self):
        """
        Utility function that makes a shallow copy of the list holding analysis data structures.
        """
        return self.analysis_results[:]

    def recordings_copy(self):
        """
        Utility function that makes a shallow copy of the list holding recordings.
        """
        return self.block.segments[:]

    def fromDataStoreView(self):
        """
        Returns a empty DSV that is linked to the same `.DataStore` as this DSV.
        """
        return DataStoreView(ParameterSet({}), self.full_datastore)

    def print_content(self, full_recordings=False, full_ADS=False):
        """
        Prints the content of the data store (specifically the list of recordings and ADSs in the DSV).
        
        If the 
        
        Parameters
        ----------
            full_recordings : bool (optional)
                            If True each contained recording will be printed.
                            Otherwise only the overview of the recordings based on stimulus type will be shown.
                            
            full_ADS : bool (optional)
                     If True each contained ADS will be printed (for each this will print the set of their mozaik parameters together with their values).
                     Otherwise only the overview of the ADSs based on their identifier will be shown.
        """
        logger.info("DSV info:")
        logger.info("   Number of recordings: " + str(len(self.block.segments)))
        d = {}
        for st in [s.annotations['stimulus'] for s in self.block.segments]:
            d[MozaikParametrized.idd(st).name] = d.get(MozaikParametrized.idd(st).name, 0) + 1

        for k in d.keys():
            logger.info("     " + str(k) + " : " + str(d[k]))

        logger.info("   Number of ADS: " + str(len(self.analysis_results)))
        d = {}
        for ads in self.analysis_results:
            d[ads.identifier] = d.get(ads.identifier, 0) + 1

        for k in d.keys():
            logger.info("     " + str(k) + " : " + str(d[k]))

        if full_recordings:
            logger.info('RECORDING RESULTS')
            for s in [s.annotations['stimulus'] for s in self.block.segments]:
                logger.info(str(s))

        if full_ADS:
            logger.info('ANALYSIS RESULTS')
            for a in self.analysis_results:
                logger.info(str(a))
    
    def __add__(self, other):
        new_dsv = self.fromDataStoreView()
        assert len(set(self.block.segments)) == len(self.block.segments)
        assert len(set(other.block.segments)) == len(other.block.segments)
        assert len(set(self.analysis_results)) == len(self.analysis_results)
        assert len(set(other.analysis_results)) == len(other.analysis_results)
        
        new_dsv.block.segments = list(set(self.block.segments) | set(other.block.segments))
        new_dsv.analysis_results = list(set(self.analysis_results) | set(other.analysis_results))
        new_dsv.sensory_stimulus = self.sensory_stimulus_copy()
        new_dsv.sensory_stimulus.update(other.sensory_stimulus)
        return new_dsv
    
    def remove_ads_from_datastore(self):
        """
        This operation removes all ADS that are present in this DataStoreView from the master DataStore.
        """
        if self.full_datastore == self:
           self.analysis_results = []
        else:
            for ads in self.analysis_results:
                self.full_datastore.analysis_results.remove(ads)

    def remove_ads_outside_of_dsv(self):
        if self.full_datastore != self:
            z = [ads for ads in self.full_datastore.analysis_results if ads not in self.analysis_results]
            for ads in z:
                self.full_datastore.analysis_results.remove(ads)
        
               
        
    
class DataStore(DataStoreView):
    """
    Abstract DataStore class that declares the *mozaik* data store interface.
    
    The role of mozaik data store is to store the recordings from simulation 
    (generally this means the spike trains and the various analog signals such as conductances or membrane potential),
    the analysis results and the various metedata that is generated during the model setup and it's subsequent simulation.
    
    The recordings are send to the DataStore in the neo format and are expected to be returned in neo format as well.
    
    mozaik generetas one neo segment per each model sheet (see :class:`.mozaik.sheets.Sheet`) for each presented stimulus,
    that is stored in the :class:`.DataStore`.
    
    Parameters
    ----------
    load : bool
         If False datastore will be created in the parameter.root_directory directory. 
         If True it will be loaded from the parameter.root_directory directory. 
    
    parameters : ParameterSet
               The required parameter set.
    """

    required_parameters = ParameterSet({
        'root_directory': str,
        'store_stimuli' : bool,
    })

    def __init__(self, load, parameters, **params):
        """
        Just check the parameters, and load the data.
        """
        DataStoreView.__init__(self, parameters, self, **params)

        # used as a set to quickly identify whether a stimulus was already presented
        # stimuli are otherwise saved with segments within the block as annotations
        self.stimulus_dict = collections.OrderedDict()

        # load the datastore
        if load:
            self.load()

    def set_neuron_positions(self, neuron_positions):
        self.block.annotations['neuron_positions'] = neuron_positions

    def set_neuron_annotations(self, neuron_annotations):
        self.block.annotations['neuron_annotations'] = neuron_annotations

    def set_neuron_ids(self, neuron_ids):
        self.block.annotations['neuron_ids'] = neuron_ids
        
    def set_model_parameters(self,parameters):
        self.block.annotations['model_parameters'] = parameters

    def set_sheet_parameters(self,parameters):
        self.block.annotations['sheet_parameters'] = parameters
        
    def set_experiment_parametrization_list(self,experiment_parameter_list):
        """
        The experiment_parameter_list is epected to be a list of tuples (experimenta_class,parameter_set) where
        *experiment_class* is the class of the experiment, and parameter_set is a ParameterSet instance converted to string 
        that corresponds to the parameters of the given experiment.
        """
        self.block.annotations['experiment_parameters'] = experiment_parameter_list
        
    def identify_unpresented_stimuli(self, stimuli):
        """
        This method filters out from a list of stimuli all those which have already been
        presented.
        """
        unpresented_stimuli_indexes = []
        for i,s in enumerate(stimuli):
            if not str(s) in self.stimulus_dict:
                unpresented_stimuli_indexes.append(i)
        return unpresented_stimuli_indexes

    def load(self):
        """
        The DataStore interface function to be implemented by a given backend. 
        It should load the datastore.
        """
        raise NotImplementedError

    def save(self):
        """
        The DataStore interface function to be implemented by a given backend. 
        It should store the datastore.
        """
        
        raise NotImplementedError


    def add_recording(self, segments, stimulus):
        """
        Add a recording into the datastore.
        """

        # we get recordings as seg
        for s in segments:
            s.annotations['stimulus'] = str(stimulus)
            self.block.segments.append(MozaikSegment(s))
        self.stimulus_dict[str(stimulus)] = True

    def add_null_recording(self, segments,stimulus):
        """
        Add recordings due to the null stimuli presented between the standard stimuli.
        """
        # we get recordings as seg
        for s in segments:
            s.null = True
            s.annotations['stimulus'] = str(stimulus)
            self.block.segments.append(MozaikSegment(s,null=True))

    def add_stimulus(self, data, stimulus):
        """
        The DataStore interface function that adds a stimulus into the datastore.
        """
        if self.parameters.store_stimuli:
           self._add_stimulus(data, stimulus)

    def _add_stimulus(self, data, stimulus):
        """
        This function adds raw sensory stimulus data that have been presented to the model into datastore. 
        """
        self.sensory_stimulus[str(stimulus)] = data

    def add_analysis_result(self, result):
        """
        Add analysis results to data store. If there already exists ADS in the data store with the same parametrization this operation will fail.
        """
        flag = True
        for i,ads in enumerate(self.analysis_results):
            if result.equalParams(ads):
                flag = False
                break
        
        if flag:
            self.analysis_results.append(result)
            return
        else:
            if self.replace:
               logger.info("Warning: ADS with the same parametrization already added in the datastore.: %s" % (str(result))) 
               self.analysis_results[i] = result
               return
            logger.error("Analysis Data Structure with the same parametrization already added in the datastore. Currently uniqueness is required. The ADS was not added. User should modify analysis specification to avoid this!: \n %s \n %s " % (str(result),str(ads)))
            raise ValueError("Analysis Data Structure with the same parametrization already added in the datastore. Currently uniqueness is required. The ADS was not added. User should modify analysis specification to avoid this!: %s \n %s" % (str(result),str(ads)))

class Hdf5DataStore(DataStore):
    """
    An DataStore that saves all it's data in a hdf5 file and an associated
    analysis results file, which just becomes the pickled self.analysis_results
    dictionary.
    """
    def load(self):
        #load the data
        iom = NeoHdf5IO(filename=self.parameters.root_directory + '/datastore.hdf5')
        self.block = iom.get('/block_0')

        # re-wrap segments
        new = []
        for s in self.block.segments:
            new.append(MozaikSegment(s))

        self.block.segments = new

        #now just construct the stimulus dictionary
        for s in self.block.segments:
            self.stimulus_dict[s.stimulus] = True

        f = open(self.parameters.root_directory + '/datastore.analysis.pickle', 'rb')
        self.analysis_results = cPickle.load(f)

    def save(self):
        # we need to first unwrap segments from MozaikWrapper
        old = self.block.segments[:]
        self.block.segments = []
        for s in old:
            self.block.segments.append(s.original_segment)

        #save the recording itself
        iom = NeoHdf5IO(filename=self.parameters.root_directory + '/datastore.hdf5')
        iom.write_block(self.block)

        # put back wrapped segments
        self.block.segments = old

        f = open(self.parameters.root_directory + '/datastore.analysis.pickle', 'wb')
        cPickle.dump(self.analysis_results, f)
        f.close()

    def add_analysis_result(self, result):
        flag = True
        for i,ads in enumerate(self.analysis_results):
            if result.equalParams(ads):
                flag = False
                break
        
        if flag:
            self.analysis_results.append(result)
            return
        else:
            if self.replace:
               logger.info("Warning: ADS with the same parametrization already added in the datastore.: %s" % (str(result))) 
               self.analysis_results[i] = result
               return
            logger.error("Analysis Data Structure with the same parametrization already added in the datastore. Currently uniqueness is required. The ADS was not added. User should modify analysis specification to avoid this!: \n %s \n %s " % (str(result),str(ads)))
            raise ValueError("Analysis Data Structure with the same parametrization already added in the datastore. Currently uniqueness is required. The ADS was not added. User should modify analysis specification to avoid this!: %s \n %s" % (str(result),str(ads)))



class PickledDataStore(Hdf5DataStore):
    """
    An DataStore that saves all it's data as a simple pickled files
    """

    def load(self):

        f = open(self.parameters.root_directory + '/datastore.recordings.pickle',  'rb')
        self.block = cPickle.load(f)
        for s in self.block.segments:
            s.full = False
            s.datastore_path = self.parameters.root_directory

        if os.path.isfile(self.parameters.root_directory + '/datastore.analysis.pickle'):
            f = open(self.parameters.root_directory + '/datastore.analysis.pickle', 'rb')
            self.analysis_results = cPickle.load(f)
        else:
            self.analysis_results = []
        
        if os.path.isfile(self.parameters.root_directory + '/datastore.sensory.stimulus.pickle'):    
            f = open(self.parameters.root_directory + '/datastore.sensory.stimulus.pickle', 'rb')
            self.sensory_stimulus = cPickle.load(f)
        else:
            self.sensory_stimulus = {}

    def save(self):
        f = open(self.parameters.root_directory + '/datastore.recordings.pickle', 'wb')
        cPickle.dump(self.block, f)
        f.close()

        f = open(self.parameters.root_directory + '/datastore.analysis.pickle', 'wb')
        cPickle.dump(self.analysis_results, f)
        f.close()

        f = open(self.parameters.root_directory + '/datastore.sensory.stimulus.pickle', 'wb')
        cPickle.dump(self.sensory_stimulus, f)
        f.close()

    def add_recording(self, segments, stimulus):
        # we get recordings as seg
        for s in segments:
            s.annotations['stimulus'] = str(stimulus)
            self.block.segments.append(
                PickledDataStoreNeoWrapper(s,
                                           'Segment' + str(len(self.block.segments)),
                                           self.parameters.root_directory))
            f = open(self.parameters.root_directory + '/' + 'Segment'
                     + str(len(self.block.segments) - 1) + ".pickle", 'wb')
            cPickle.dump(s, f)

        self.stimulus_dict[str(stimulus)] = True


    def add_null_recording(self, segments,stimulus):
        """
        Add recordings due to the null stimuli presented between the standard stimuli.
        """
        # we get recordings as seg
        for s in segments:
            s.annotations['stimulus'] = str(stimulus)
            self.block.segments.append(
                PickledDataStoreNeoWrapper(s,
                                           'Segment' + str(len(self.block.segments)),
                                           self.parameters.root_directory,null=True))
            f = open(self.parameters.root_directory + '/' + 'Segment'
                     + str(len(self.block.segments) - 1) + ".pickle", 'wb')
            cPickle.dump(s, f)
