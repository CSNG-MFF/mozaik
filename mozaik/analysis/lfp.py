# -*- coding: utf-8 -*-
"""
Module containing lfp specific analysis.
"""
import mozaik
import numpy
import quantities as qt
from .analysis import Analysis
from mozaik.tools.mozaik_parametrized import colapse, colapse_to_dictionary, MozaikParametrized
from mozaik.analysis.data_structures import PerAreaAnalogSignalList
from mozaik.analysis.helper_functions import psth
from parameters import ParameterSet
from mozaik.storage import queries
from mozaik.tools.circ_stat import circ_mean, circular_dist
from mozaik.tools.neo_object_operations import neo_mean, neo_sum
from builtins import zip
from collections import OrderedDict
from mozaik.tools.distribution_parametrization import PyNNDistribution
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal

logger = mozaik.getMozaikLogger()

class LFPFromSynapticCurrents(Analysis):
    """
    This analysis takes each recording in DSV that has been done in response to stimulus type 'stimulus_type'
    and calculates the LFP signal using a linear combination of excitatory and inhibitory synaptic currents as a proxy.
    For each set of equal recordings (except trial) it creates one PerAreaAnalogSignalList
    `AnalysisDataStructure` instance containing the LFP signal calculated on each sub-area of the cortical space,
    defined through the x_coords and y_coords parameters.


    Other parameters
    ----------------
    bin_length : float (ms)
               the size of bin to construct the lfp signal from

    points_distance : float (micrometers)
             The distance separating each spatial points around which the LFPs will be calculated

    side_length : float (micrometers)
             The length of the side of the squares in which the LFPs will be calculated

    """
    required_parameters = ParameterSet({
      'bin_length' : float,  # the bin length of the PSTH
      'points_distance' : float,  # the bin length of the PSTH
      'side_length' : float,  # the bin length of the PSTH
    })

    # Need temoral bin, coordinates, and squares
    def perform_analysis(self):
        for sheet in self.datastore.sheets():

            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet)
            segs = dsv1.get_segments()
            analog_ids = segs[0].get_stored_esyn_ids()
            st = [MozaikParametrized.idd(s) for s in dsv1.get_stimuli()]
            logger.info(eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet])
            sx = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sx']
            sy = eval(dsv1.full_datastore.block.annotations['sheet_parameters'])[sheet]['sy']

            for seg, s in zip(segs, st):
                x_axis = numpy.arange(-sx/2 + self.parameters.points_distance/2, sx/2 + self.parameters.points_distance/2, self.parameters.points_distance)
                y_axis = numpy.arange(-sy/2 + self.parameters.points_distance/2, sy/2 + self.parameters.points_distance/2, self.parameters.points_distance)
                t_axis = numpy.arange(0, s.duration, self.parameters.bin_length)
                ase = seg.get_esyn(analog_ids[0])
                asv = seg.get_vm(analog_ids[0])
                null_signal = NeoAnalogSignal(numpy.zeros(len(ase)),
                            t_start=ase.t_start,
                            sampling_period=ase.sampling_period,
                            units=(ase[0] * asv[0]).units)

                m = [[null_signal for _ in x_axis] for _ in y_axis]
                for aid in analog_ids:
                    sid = self.datastore.get_sheet_indexes(sheet,aid)
                    x = dsv1.get_neuron_positions()[sheet][0][sid]
                    y = dsv1.get_neuron_positions()[sheet][1][sid]
                    x_id = int((x + sx/2 + self.parameters.points_distance/2)/self.parameters.points_distance)
                    y_id = int((y + sy/2 + self.parameters.points_distance/2)/self.parameters.points_distance)
                    e_syn = seg.get_esyn(aid)
                    i_syn = seg.get_isyn(aid)
                    vm = seg.get_vm(aid)
                    time_step = e_syn.sampling_period
                    idiff = int(6/time_step)
                    padding = NeoAnalogSignal(numpy.zeros(idiff),
                            t_start=vm.t_start,
                            sampling_period=time_step,
                            units=(vm[0] * e_syn[0]).units)
                    i_curr = (vm * i_syn)[idiff:]
                    scalar = NeoAnalogSignal(numpy.array([1.65]*(len(vm) - idiff)),
                            t_start=vm.t_start,
                            sampling_period=time_step,
                            units=qt.dimensionless)
                    lfp = ((vm * e_syn)[:-idiff] + scalar * i_curr.time_shift(-i_curr.t_start)).time_shift(i_curr.t_start)
                    m[x_id][y_id] += lfp.concatenate(padding)

                self.datastore.full_datastore.add_analysis_result(
                    PerAreaAnalogSignalList(m,x_axis,y_axis,lfp.units,
                                   stimulus_id=str(s),
                                   x_axis_name='time',
                                   y_axis_name='LFP',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))


class LFPFromSpikes(Analysis):
    """
    This analysis takes each recording in DSV that has been done in response to stimulus type 'stimulus_type'
    and calculates the LFP signal using spikes as a proxy. For each set of equal recordings (except trial) it creates one PerAreaAnalogSignalList
    `AnalysisDataStructure` instance containing the LFP signal calculated on each sub-area of the cortical space,
    defined through the x_coords and y_coords parameters.


    Other parameters
    ----------------
    bin_length : float (ms)
               the size of bin to construct the lfp signal from

    x_coords : list of float (micrometers)
             The x coordinates of the spatial points around which the LFPs will be calculated

    y_coords : list of float (micrometers)
             The y coordinates of the spatial points around which the LFPs will be calculated

    points_distance : float (micrometers)
             The distance separating each spatial points around which the LFPs will be calculated

    side_length : float (micrometers)
             The length of the side of the squares in which the LFPs will be calculated

    """
    required_parameters = ParameterSet({
      'bin_length' : float,  # the bin length of the PSTH
      'points_distance' : float,  # the bin length of the PSTH
      'side_length' : float,  # the bin length of the PSTH
    })

    # Need temoral bin, coordinates, and squares
    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv1 = queries.param_filter_query(self.datastore, sheet_name=sheet)
            segs = dsv1.get_segments()
            st = [MozaikParametrized.idd(s) for s in dsv1.get_stimuli()]
            mean_rates = [numpy.array(s.mean_rates()) for s in segs]

            logger.debug('Adding PerNeuronValue containing trial averaged firing rates to datastore')
            for mr, vr, st in zip(_mean_rates,_var_rates, s):

                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(mr,segs[0].get_stored_spike_train_ids(),units,
                                   stimulus_id=str(st),
                                   value_name='Firing rate',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))

                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(vr,segs[0].get_stored_spike_train_ids(),units,
                                   stimulus_id=str(st),
                                   value_name='Tria-to-trial Var of Firing rate',
                                   sheet_name=sheet,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))
