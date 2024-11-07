"""
This module contains special analysis functions that relate to some tehnical mozaik architecture aspects and thus 
do not represent a standard analysis
"""
from mozaik.analysis.data_structures import PerNeuronValue
from mozaik.analysis.analysis import Analysis
from mozaik.storage import queries
from parameters import ParameterSet
from mozaik.tools.mozaik_parametrized import MozaikParametrized
import quantities as qt
import numpy
import mozaik
import pickle

logger = mozaik.getMozaikLogger()
from mozaik.controller import Global

class NeuronAnnotationsToPerNeuronValues(Analysis):
    """
    Creates a PerNeuronValues analysis data structure per each neuron
    annotation that is defined for all neurons in a given sheet.

    This analysis is aware of several mozaik specific annotations and adds additional
    appropriate information to the PerNeuronValue ADS (i.e. setting period to
    numpy.pi of orientation preference of initial connection fields).
    Users are expected to modify this class to add additional information for
    their new annotations if required.
    It is assumed that in future the handling of parameters around Mozaik
    might be enhanced and unified further to avoid extension of this class.
    """


    def perform_analysis(self):
        logger.info('Starting NeuronAnnotationsToPerNeuronValues Analysis')
        anns = self.datastore.get_neuron_annotations()

        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore,sheet_name=sheet)
            keys = set([])

            for n in range(0, len(anns[sheet])):
                keys = keys.union(anns[sheet][n].keys())

            for k in keys:
                # first check if the key is defined for all neurons
                key_ok = True

                for n in range(0, len(anns[sheet])):
                    if not k in anns[sheet][n]:
                        key_ok = False
                        break

                if key_ok:
                    values = []
                    for n in range(0, len(anns[sheet])):
                        values.append(anns[sheet][n][k])

                    period = None
                    if k == 'LGNAfferentOrientation':
                        period = numpy.pi
                    if k == 'LGNAfferentPhase':
                        period = 2*numpy.pi

                    self.datastore.full_datastore.add_analysis_result(
                        PerNeuronValue(values,
                                       dsv.get_sheet_ids(sheet),
                                       qt.dimensionless,
                                       period=period,
                                       value_name=k,
                                       sheet_name=sheet,
                                       tags=self.tags,
                                       analysis_algorithm=self.__class__.__name__))


class SummarizeSingleValues(Analysis):


    required_parameters = ParameterSet({
        'file_name': str,  # the first value name 
    })


    def perform_analysis(self):

        dsv = queries.param_filter_query(self.datastore,identifier='SingleValue')

        f = open(Global.root_directory+self.parameters.file_name,'w')

        for a in dsv.get_analysis_result():
            f.write("%s %s %s %s %s\n" % (a.sheet_name, a.value_name, str(a.value), a.analysis_algorithm, a.stimulus_id))
        f.close()

class ExportRawSpikeData(Analysis):
    """
    Exports raw data from the simulation in a simple numpy readable format.
    It will export responses to all stimuli, and for each stimulus the measurments 
    specified in the `variables_to_export` will be saved.

    The export format is as follows:

    At the top level there is a dictionary with each key corresponding to one of the stimuli in the datastore that was exported.
    The stringified `MozaikParamerised.idd` of the stimulus is used as the key.

    Each entry in the dictionary is a dictionary with keys corresponding to the names of populations in the model.

    Each entry in this dictionary then contains 3D ndarray. The first dimension corresponds to neurons in the given 
    sheet, the second dimension to repeated trials of the stimulus presentation, the third dimension has variable 
    length and contains the spike times emitted by the given neuron on the given trial.

    This structure will be saved in a .npy format in file named `file_name`.
    """

    required_parameters = ParameterSet({
        'file_name' : str
    })

    def perform_analysis(self):
        res = {}

        dsvs = queries.partition_by_stimulus_paramter_query(self.datastore,["trial"])

        for dsv in dsvs:
                # lets get rid of trial from the stimulus ID to use for the new ADS
                stim = dsv.get_stimuli()[0]
                stim = MozaikParametrized.idd(stim)
                stim.trial = None
                stim = str(stim)

                if not stim in res.keys():
                       res[stim]={}

                for seg in self.datastore.get_segments():
                    sheet_name = seg.annotations['sheet_name']

                    if sheet_name not in res[stim].keys():
                       res[stim][sheet_name]=[] 

                    res[stim][sheet_name].append([s.magnitude for s in seg.spiketrains]) 

        for k in res.keys():
            for kk in res[k].keys():
                res[k][kk] = numpy.array(res[k][kk])

        f = open(Global.root_directory+self.parameters.file_name,'wb')
        pickle.dump(res,f)