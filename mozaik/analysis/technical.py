"""
This module contains special analysis functions that relate to some tehnical mozaik architecture aspects and thus 
do not represent a standard analysis
"""
from mozaik.analysis.data_structures import PerNeuronValue
from mozaik.analysis.analysis import Analysis
from mozaik.storage import queries
from parameters import ParameterSet
from sets import Set
import quantities as qt
import numpy
import mozaik

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
            keys = Set([])

            for n in xrange(0, len(anns[sheet])):
                keys = keys.union(anns[sheet][n].keys())

            for k in keys:
                # first check if the key is defined for all neurons
                key_ok = True

                for n in xrange(0, len(anns[sheet])):
                    if not k in anns[sheet][n]:
                        key_ok = False
                        break

                if key_ok:
                    values = []
                    for n in xrange(0, len(anns[sheet])):
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
