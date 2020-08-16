"""
This module contains various utility functions often useb by analysis code.
"""

import numpy
import quantities as qt
import mozaik
import mozaik.tools.units as munits
from mozaik.controller import Global
from mozaik.storage.queries import *
from mozaik.tools.mozaik_parametrized import *
from neo import AnalogSignal

logger = mozaik.getMozaikLogger()


def psth(spike_list, bin_length, normalize=True):
    """
    The function returns the psth of the spiketrains with bin length bin_length.
    
    Parameters
    ----------
    spike_list : list(SpikeTrain )
               The list of spike trains. They are assumed to start and end at the same time.

    bin_length : float (ms) 
               Bin length.
               
    normalized : bool
               If true the psth will return the instantenous firing rate, if False it will return spike count per bin. 

    Returns
    -------
    psth : AnalogSignal
           The PSTH of the spiketrain. 
    
    Note
    ----
    The spiketrains are assumed to start and stop at the same time!
    """
    t_start = round(spike_list[0].t_start.rescale(qt.ms), 5)
    t_stop = round(spike_list[0].t_stop.rescale(qt.ms), 5)
    num_bins = int(round((t_stop - t_start) / bin_length))
    r = (float(t_start), float(t_stop))

    for sp in spike_list:
        assert len(numpy.histogram(sp, bins=num_bins, range=r)[0]) == num_bins

    normalizer = 1.0
    if normalize:
        normalizer = bin_length / 1000.0

    h = [
        AnalogSignal(
            numpy.histogram(sp, bins=num_bins, range=r)[0] / normalizer,
            t_start=t_start * qt.ms,
            sampling_period=bin_length * qt.ms,
            units=munits.spike_per_sec,
        )
        for sp in spike_list
    ]
    return h


def psth_across_trials(spike_trials, bin_length):
    """
    It returns PSTH averaged across the spiketrains
    
    
    Parameters
    ----------
    spike_trials : list(list(SpikeTrain)) 
                 should contain a list of lists of neo spiketrains objects,
                 first coresponding to different trials and second to different neurons.
                 The function returns the histogram of spikes across trials with bin length
                 bin_length.
   
    Returns
    -------
    psth : AnalogSignal
         The PSTH averages over the spiketrains in spike_trials.
                 
    Note
    ----
    The spiketrains are assumed to start and stop at the same time.
    """
    return sum([psth(st, bin_length) for st in spike_trials]) / len(spike_trials)


def pnv_datastore_view_to_tensor(pnv_view):
    """
    Assuming:
        * the pnv_view contains a set of identical pnvs to identical stimuli with the exception of variation of some stimulus parameters
        * all values of the varied parameters lay on a n-dimensional grid where n is the number of parameters
    This function turns the pnv_view set into a n+1 dimensional tensor of values, where the first n dimensions correspond to the stimulus 
    parameters varied, and nth+1 dimension correspond to the number of neurons recorded in the pnvs in the pnv_view.    
    """
    assert queries.equal_ads(
        pnv_view, except_params=["stimulus_id"]
    ), "All ADS in the view have to be same with the exception of stimulus"
    pnvs = dsv.get_analysis_result()
    assert (
        pnvs[0].identifier == "PerNeuronValue"
    ), "All ADS have to be of the PerNeuronValue type"
    assert queries.ads_with_equal_stimulus_type(
        pnv_view, allow_None=True
    ), "All PNVs have to be with respect to the same stimulus type"

    # let's find out which parameters are varying
    params = varying_parameters(MozaikParametrized.idd(pnv.stimulus_id) for pnv in pnvs)

    # let's determine what should be coordinates along the different parameter axes
    _min = []
    _max = []
    _step = []
    for p in params:
        s = set([getattr(pnv, p) for pnv in pnv_view]).sort()
        _min.append(s[0])
        _max.append(s[-1])
        _step.append(s[1] - s[0])

    # let's verify that each pnv sits on the determined coordinates
    coords = [
        numpy.arange(mmin, mmax, step) for (mmin, mmax, step) in zip(_min, _max, _step)
    ]

    for pnv in pnvs:
        for p, c in zip(params, coords):
            assert getattr(pnv, p) in c, (
                "Value %f of parameter < %s > of PNV[%s] does not conform to coordinate range"
                % (getattr(pnv, p), p, str(pnv), str(c))
            )

    # let's create the tensor
    tensor = numpy.empty([len(c) for c in coords] + [len(pnvs[0].values)])
    tensor.fill(numpy.nan)

    # let's insert data into the tensor
    for pnv in pnvs:
        coord = [getattr(pnv, p) for p in params]
        tensor[coord] = pnv.values

    return (tensor, params)


def pnv_datastore_view_to_tensor(pnv_view, allow_missing=False, pickle_file=None):
    """
    Assuming:
        * the pnv_view contains a set of identical pnvs to identical stimuli with the exception of variation of some stimulus parameters
        * all values of the varied parameters lay on a n-dimensional (irregularly spaced) grid where n is the number of parameters
    This function turns the pnv_view set into a n+1 dimensional tensor of values, where the first n dimensions correspond to the stimulus 
    parameters varied, and nth+1 dimension correspond to the number of neurons recorded in the pnvs in the pnv_view.    
    """
    assert equal_ads(
        pnv_view, except_params=["stimulus_id"]
    ), "All ADS in the view have to be same with the exception of stimulus"
    pnvs = pnv_view.get_analysis_result()
    assert (
        pnvs[0].identifier == "PerNeuronValue"
    ), "All ADS have to be of the PerNeuronValue type"
    assert ads_with_equal_stimulus_type(
        pnv_view, allow_None=True
    ), "All PNVs have to be with respect to the same stimulus type"

    # let's find out which parameters are varying
    params = varying_parameters(
        [MozaikParametrized.idd(pnv.stimulus_id) for pnv in pnvs]
    )

    # let's determine what should be coordinates along the different parameter axes
    coords = []
    for p in params:
        s = numpy.sort(
            numpy.unique(
                [getattr(MozaikParametrized.idd(pnv.stimulus_id), p) for pnv in pnvs]
            )
        )
        coords.append(s)

    if allow_missing:
        assert numpy.prod([len(c) for c in coords]) >= len(pnvs)
    else:
        assert numpy.prod([len(c) for c in coords]) == len(pnvs)

    for pnv in pnvs:
        for p, c in zip(params, coords):
            assert getattr(MozaikParametrized.idd(pnv.stimulus_id), p) in c, (
                "Value %f of parameter <%s> of PNV[%s] does not conform to coordinate range: %s"
                % (
                    getattr(MozaikParametrized.idd(pnv.stimulus_id), p),
                    p,
                    str(pnv),
                    str(c),
                )
            )

    # let's create the tensor
    tensor = numpy.empty([len(c) for c in coords] + [len(pnvs[0].values)])
    tensor.fill(numpy.nan)

    # let's insert data into the tensor
    for pnv in pnvs:
        co = [getattr(MozaikParametrized.idd(pnv.stimulus_id), p) for p in params]
        idxs = numpy.array(
            [numpy.where(coord == c)[0][0] for coord, c in zip(coords, co)]
        )
        tensor[tuple(idxs)] = pnv.values

    # pickle save
    if pickle_file:
        import pickle

        f = open(Global.root_directory + "/" + pickle_file, "wb")
        pickle.dump((tensor, params, coords, pnvs[0].ids), f)
        f.close()

    return (tensor, params, coords, pnvs[0].ids)
