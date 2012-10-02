"""
docstring goes here
"""

import mozaik.storage.queries as queries
from mozaik.visualization.plotting import (Plotting, RasterPlot, GSynPlot,
                                           VmPlot, ConductanceSignalListPlot,
                                           AnalogSignalListPlot,
                                           RetinalInputMovie)
from NeuroTools.parameters import ParameterSet, ParameterDist
import matplotlib.gridspec as gridspec
from mozaik.storage.queries import (select_stimuli_type_query,
                                    select_result_sheet_query,
                                    partition_by_stimulus_paramter_query)

from simple_plot import SpikeRasterPlot, SpikeHistogramPlot

import pylab


class Figure2Gratings(Plotting):
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'neuron': int,  # which neuron to show
    })

    def subplot(self, subplotspec, params):
        gs = gridspec.GridSpecFromSubplotSpec(12, 18, subplot_spec=subplotspec,
                                              hspace=1.0, wspace=1.0)

        dsv = queries.select_stimuli_type_query(self.datastore,
                                                'FullfieldDriftingSinusoidalGrating',
                                                {'orientation': 0.0,
                                                 'max_luminance': 90.0})

        print len(dsv.get_segments())

        lgn_on_dsv = queries.select_result_sheet_query(dsv, 'X_ON')
        lgn_off_dsv = queries.select_result_sheet_query(dsv, 'X_OFF')
        lgn_spikes = [[s.spiketrains for s in lgn_on_dsv.get_segments()],
                      [s.spiketrains for s in lgn_off_dsv.get_segments()]]

        SpikeRasterPlot(lgn_spikes, neurons=[0], x_axis=False, x_label=None,
                        colors=['#FACC2E', '#0080FF'])(gs[1:4, 0:5])
        SpikeHistogramPlot(lgn_spikes, neurons=[0], x_axis=False, x_label=None,
                           colors=['#FACC2E', '#0080FF'])(gs[4:5, 0:5])
        SpikeRasterPlot(lgn_spikes, neurons=[5], x_axis=False, x_label=None,
                        colors=['#FACC2E', '#0080FF'])(gs[7:10, 0:5])
        SpikeHistogramPlot(lgn_spikes, neurons=[5],
                           colors=['#FACC2E', '#0080FF'])(gs[10:11, 0:5])

        dsv1 = queries.select_result_sheet_query(dsv,
                                                 self.parameters.sheet_name)
        SpikeRasterPlot([[s.spiketrains for s in dsv1.get_segments()]],
                        neurons=[self.parameters.neuron], x_axis=False,
                        x_label=None)(gs[:3, 6:14])
        SpikeHistogramPlot([[s.spiketrains for s in dsv1.get_segments()]],
                           neurons=[self.parameters.neuron], x_axis=False,
                           x_label=None)(gs[3:4, 6:14])

        p = params.copy()
        p.setdefault('title', None)
        p.setdefault('x_axis', None)
        p.setdefault('x_label', None)
        VmPlot(dsv, ParameterSet({'sheet_name': self.parameters.sheet_name,
                                  'neuron': self.parameters.neuron})).subplot(gs[4:8, 6:14], p)
        p = params.copy()
        p.setdefault('title', None)
        GSynPlot(dsv, ParameterSet({'sheet_name': self.parameters.sheet_name,
                                    'neuron': self.parameters.neuron})).subplot(gs[8:12, 6:14], p)

        ConductanceSignalListPlot(queries.TagBasedQuery(
                                        ParameterSet({'tags': ['GSTA']})
                                      ).query(self.datastore),
                                  ParameterSet({'sheet_name': self.parameters.sheet_name,
                                                'normalize_individually': True})
                                      ).subplot(gs[7:10, 15:],
                                  params.copy())

        p = params.copy()
        p.setdefault('mean', False)
        AnalogSignalListPlot(dsv, ParameterSet({'sheet_name': self.parameters.sheet_name,
                                                'ylabel': 'AC (norm)'})).subplot(gs[2:5, 15:], p)


class Figure2NaturalImagesWithEyeMovement(Plotting):
    required_parameters = ParameterSet({
        'sheet_name': str,  # the name of the sheet for which to plot
        'neuron': int,  # which neuron to show
    })

    def subplot(self, subplotspec, params):
        gs = gridspec.GridSpecFromSubplotSpec(12, 18, subplot_spec=subplotspec,
                                              hspace=1.0, wspace=1.0)

        dsv = queries.select_stimuli_type_query(self.datastore,
                                                'NaturalImageWithEyeMovement',
                                                {})

        print len(dsv.get_segments())

        lgn_on_dsv = queries.select_result_sheet_query(dsv, 'X_ON')
        lgn_off_dsv = queries.select_result_sheet_query(dsv, 'X_OFF')
        lgn_spikes = [[s.spiketrains for s in lgn_on_dsv.get_segments()],
                      [s.spiketrains for s in lgn_off_dsv.get_segments()]]

        SpikeRasterPlot(lgn_spikes, neurons=[0], x_axis=False, x_label=None,
                        colors=['#FACC2E', '#0080FF'])(gs[1:4, 0:5])
        SpikeHistogramPlot(lgn_spikes, neurons=[0], x_axis=False, x_label=None,
                           colors=['#FACC2E', '#0080FF'])(gs[4:5, 0:5])
        SpikeRasterPlot(lgn_spikes, neurons=[5], x_axis=False, x_label=None,
                        colors=['#FACC2E', '#0080FF'])(gs[7:10, 0:5])
        SpikeHistogramPlot(lgn_spikes, neurons=[5],
                           colors=['#FACC2E', '#0080FF'])(gs[10:11, 0:5])

        dsv1 = queries.select_result_sheet_query(dsv, self.parameters.sheet_name)
        SpikeRasterPlot([[s.spiketrains for s in dsv1.get_segments()]],
                        neurons=[self.parameters.neuron], x_axis=False,
                        x_label=None)(gs[:3, 6:14])
        SpikeHistogramPlot([[s.spiketrains for s in dsv1.get_segments()]],
                           neurons=[self.parameters.neuron], x_axis=False,
                           x_label=None)(gs[3:4, 6:14])

        p = params.copy()
        p.setdefault('title', None)
        p.setdefault('x_axis', None)
        p.setdefault('x_label', None)
        VmPlot(dsv,
               ParameterSet({'sheet_name': self.parameters.sheet_name,
                             'neuron': self.parameters.neuron})).subplot(gs[4:8, 6:14], p)
        p = params.copy()
        p.setdefault('title', None)
        GSynPlot(dsv,
                 ParameterSet({'sheet_name': self.parameters.sheet_name,
                               'neuron': self.parameters.neuron})).subplot(gs[8:12, 6:14], p)

        ConductanceSignalListPlot(
                queries.TagBasedQuery(ParameterSet({'tags': ['GSTA']})).query(self.datastore),
                ParameterSet({'sheet_name': self.parameters.sheet_name,
                              'normalize_individually': True})
            ).subplot(gs[7:10, 15:], params.copy())

        p = params.copy()
        p.setdefault('mean', False)
        AnalogSignalListPlot(dsv,
                             ParameterSet({'sheet_name': self.parameters.sheet_name,
                                           'ylabel': 'AC (norm)'})).subplot(gs[2:5, 15:], p)
