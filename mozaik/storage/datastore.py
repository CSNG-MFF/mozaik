"""
docstring goes here
"""
import numpy
from parameters import ParameterSet
from neo.core.block import Block
#from neo.io.hdf5io import NeoHdf5IO
import mozaik
from mozaik.framework.interfaces import MozaikParametrizeObject
from neo_neurotools_wrapper import MozaikSegment, PickledDataStoreNeoWrapper
from mozaik.tools.mozaik_parametrized import  MozaikParametrized,filter_query
import cPickle
import collections

logger = mozaik.getMozaikLogger("Mozaik")



class DataStoreView(MozaikParametrizeObject):
    """
    This class represents a subset of a DataStore and defines the query
    interface and the structure in which the data are stored in the memory of
    any datastore. Main role of this class is to allow for creating subsets of
    data stored in the DataStore, so that one can restrict other parts of
    Mozaik to work only over these subsets. This is done via means of queries
    (see storage.queries) which produce DataStoreView objects and can be
    chained to work like filters.

    Note that the actual datastores inherit from this object and define how the
    data are actualy stored on other media (i.e. HD via i.e. hdf5 file format).

    Data store should aggregate all data and analysis results collected across
    experiments and a given model.

    Experiments results storage:

    These are stored as a simple list in the self.block.segments variable. Each
    segment corresponds to recordings from a single sheet to a single stimulus.

    Analysis results storage:

    There are three categories of results that map to the MoziakLite model
    structural building blocks:

    Model specific - most general
    Sheet specific
    Neuron specific

    This is a nested structure of which each level is addressed with appropriate
    identifier and all levels can hold results. In future this structure might
    be extended with more levels depending on the native structures added to
    mozaik (i.e. cortical area, cortical layer etc.)

    Further at each level the analysis results are addressed by the
    AnalysisDataStructure identifier. The call with this specification returns
    a set of AnalysisDataStructures that correspond to the above addressing.
    This can be a list. If further more specific 'addressing' is required it
    has to be done by the corresponding visualization or analysis code that
    asked for the AnalysisDataStructure's based on the knowledge of their
    content. Or a specific query filters can be written that understand the specific type
    of AnalysisDataStructure and can filter them based on their internal data.

    DataStoreView also keeps a reference to a full Datastore <full_datastore>
    from which it was originally created (this might be have happened via a
    chain of DSVs). This is on order to allow for operations that work over DSV
    to insert their results into the original full datastore as this is
    (almost?) always the desired behaviours (note DSV does not actually have
    functions to add new recordings or analysis results).
    
    By default, the datastore will refuse to add a new AnalysisDataStructure 
    to the datastore if the new ADS has the same values of all its parameters as
    some other ADS already inserted in the datastore. This is so that each ADS
    stored in datastore is uniquely identifiable based on its parameters.
    If the datastore is created (loaded) with flag replace set to True in the situation
    of such conflict the datastore will replace the new ADS for the one already in the datastore.
    """

    def __init__(self, parameters, full_datastore,replace=False):
        """
        Just check the parameters, and load the data.
        """
        MozaikParametrizeObject.__init__(self, parameters)
        # we will hold the recordings as one neo Block
        self.block = Block()
        self.analysis_results = []
        self.replace = replace
        self.retinal_stimulus = collections.OrderedDict()
        self.full_datastore = full_datastore  # should be self if actually the
                                              # instance is actually DataStore

    def get_segments(self):
        return self.block.segments

    def sheets(self):
        """
        Returns the list of all sheets that are present in at least one of the
        segments in the given DataStoreView
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
        return self.full_datastore.block.annotations['neuron_positions']

    def get_sheet_indexes(self, sheet_name,neuron_id):
        """
        Returns the indexes of neurons in the sheet given the idds (this should be primarily used with annotations data such as positions etc.)
        """
        ids = self.full_datastore.block.annotations['neuron_ids'][sheet_name]
        if isinstance(neuron_id,list) or isinstance(neuron_id,numpy.ndarray):
          return [ids.index(i) for i in neuron_id]
        else:
          return ids.index(neuron_id)

    def get_sheet_ids(self, sheet_name,indexes=None):
        """
        Returns the idds of neurons in the sheet given the indexes (this should be primarily used with annotations data such as positions etc.)
        """
        # find first segment from sheet sheet_name
        if indexes == None:
            return self.full_datastore.block.annotations['neuron_ids'][sheet_name]
        else:
            return self.full_datastore.block.annotations['neuron_ids'][sheet_name][indexes]

    def get_neuron_annotations(self):
        return self.full_datastore.block.annotations['neuron_annotations']

    def get_stimuli(self):
        """
        Returns a list of stimuli (as strings). The order of the stimuli
        corresponds to the order of segments returned by the get_segments()
        call.
        """
        return [s.annotations['stimulus'] for s in self.block.segments]

    def get_analysis_result(self, **kwargs):
        return filter_query(self.analysis_results,**kwargs)

    def get_retinal_stimulus(self, stimuli=None):
        if stimuli == None:
            return self.retinal_stimulus.values()
        else:
            return [self.retinal_stimulus[s] for s in stimuli]

    def retinal_stimulus_copy(self):
        new_dict = collections.OrderedDict()
        for k in self.retinal_stimulus.keys():
            new_dict[k] = self.retinal_stimulus[k]
        return new_dict

    def analysis_result_copy(self):
        return self.analysis_results[:]

    def recordings_copy(self):
        return self.block.segments[:]

    def fromDataStoreView(self):
        return DataStoreView(ParameterSet({}), self.full_datastore)

    def print_content(self, full_recordings=False, full_ADS=False):
        """
        Info for debugging purposes
        """
        print "DSV info:"
        print "   Number of recordings: " + str(len(self.block.segments))
        d = {}
        for st in [s.annotations['stimulus'] for s in self.block.segments]:
            d[MozaikParametrized.idd(st).name] = d.get(MozaikParametrized.idd(st).name, 0) + 1

        for k in d.keys():
            print "     " + str(k) + " : " + str(d[k])

        print "   Number of ADS: " + str(len(self.analysis_results))
        d = {}
        for ads in self.analysis_results:
            d[ads.identifier] = d.get(ads.identifier, 0) + 1

        for k in d.keys():
            print "     " + str(k) + " : " + str(d[k])

        if full_recordings:
            print 'RECORDING RESULTS'
            for s in [s.annotations['stimulus'] for s in self.block.segments]:
                print str(s)

        if full_ADS:
            print 'ANALYSIS RESULTS'
            for a in self.analysis_results:
                print str(a)
    
class DataStore(DataStoreView):
    """
    Abstract DataStore class the declares the parameters, and enforces the
    interface
    """

    required_parameters = ParameterSet({
        'root_directory': str,
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
        
    def identify_unpresented_stimuli(self, stimuli):
        """
        It will filter out from stimuli all those which have already been
        presented.
        """
        unpresented_stimuli = []
        for s in stimuli:
            if not str(s) in self.stimulus_dict:
                unpresented_stimuli.append(s)
        return unpresented_stimuli

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def add_recording(self, segment, stimulus):
        raise NotImplementedError

    def add_analysis_result(self, result, sheet_name=None, neuron_idx=None):
        raise NotImplementedError

    def add_stimulus(self, data, stimulus):
        raise NotImplementedError


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

    def add_recording(self, segments, stimulus):
        # we get recordings as seg
        for s in segments:
            s.annotations['stimulus'] = str(stimulus)
            self.block.segments.append(MozaikSegment(s))
        self.stimulus_dict[str(stimulus)] = True

    def add_stimulus(self, data, stimulus):
        self.retinal_stimulus[str(stimulus)] = data

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
            logger.error("Analysis Data Structure with the same parametrization already added in the datastore. Currently uniqueness is required. The ADS was not added. User should modify analysis specification to avoid this!: %s" % (str(result)))
            raise ValueError("Analysis Data Structure with the same parametrization already added in the datastore. Currently uniqueness is required. The ADS was not added. User should modify analysis specification to avoid this!: %s" % (str(result)))


class PickledDataStore(Hdf5DataStore):
    """
    An DataStore that saves all it's data as a simple pickled files
    """

    def load(self):
        if False:
            f = open(self.parameters.root_directory + '/datastore.recordings.pickle',  'rb')
            self.block = cPickle.load(f)
            for s in self.block.segments:
                s.full = False
                self.stimulus_dict[s.annotations['stimulus']] = True
                s.datastore_path = self.parameters.root_directory

        f = open(self.parameters.root_directory + '/datastore.analysis.pickle', 'rb')
        self.analysis_results = cPickle.load(f)
        #f = open(self.parameters.root_directory + '/datastore.retinal.stimulus.pickle', 'rb')
        #self.retinal_stimulus = cPickle.load(f)

    def save(self):
        f = open(self.parameters.root_directory + '/datastore.recordings.pickle', 'wb')
        cPickle.dump(self.block, f)
        f.close()

        f = open(self.parameters.root_directory + '/datastore.analysis.pickle', 'wb')
        cPickle.dump(self.analysis_results, f)
        f.close()

        #f = open(self.parameters.root_directory + '/datastore.retinal.stimulus.pickle', 'wb')
        #cPickle.dump(self.retinal_stimulus, f)
        #f.close()

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
