import mozaik
from mozaik.storage.datastore import PickledDataStore
from neo.core.segment import Segment
from mozaik.storage.neo_neurotools_wrapper import (
    MozaikSegment,
    PickledDataStoreNeoWrapper,
)
from parameters import ParameterSet
import numpy
import pickle
import re
import os

logger = mozaik.getMozaikLogger()


def compare_sheets_datastores(datastores):
    """
    Returns True if all datastores contain sheets with similar names
    """
    sheets = datastores[0].block.annotations["neuron_ids"].keys()
    for i in range(1, len(datastores)):
        if not numpy.array_equal(
            datastores[i].block.annotations["neuron_ids"].keys(), sheets
        ):
            return False
    return True


def compare_neurons_ids_datastores(datastores):
    """
    Returns True if all datastores contain neurons with similar ids
    """
    sheets = datastores[0].block.annotations["neuron_ids"].keys()
    for sheet in sheets:
        ids = datastores[0].block.annotations["neuron_ids"][sheet]
        for i in range(1, len(datastores)):
            if not numpy.array_equal(
                datastores[i].block.annotations["neuron_ids"][sheet], ids
            ):
                return False
    return True


def compare_neurons_position_datastores(datastores):
    """
    Returns True if all datastores contain neurons with similar positions
    """
    sheets = datastores[0].block.annotations["neuron_positions"].keys()
    for sheet in sheets:
        positions = datastores[0].block.annotations["neuron_positions"][sheet]
        for i in range(1, len(datastores)):
            if not numpy.array_equal(
                datastores[i].block.annotations["neuron_positions"][sheet], positions
            ):
                return False
    return True


def compare_neurons_annotations_datastores(datastores):
    """
    Returns True if all datastores contain neurons with similar annotations
    """
    sheets = datastores[0].block.annotations["neuron_annotations"].keys()
    for sheet in sheets:
        annotations = datastores[0].block.annotations["neuron_annotations"][sheet]
        for i in range(1, len(datastores)):
            if not numpy.array_equal(
                datastores[i].block.annotations["neuron_annotations"][sheet],
                annotations,
            ):
                return False
    return True


def merge_experiment_parameters_datastores(datastores):
    """
    Merge the experiment parameters of the datastores
    """
    experiment_parameters = []
    for datastore in datastores:
        for parameters in datastore.block.annotations["experiment_parameters"]:
            experiment_parameters.append(parameters)
    return experiment_parameters


def merge_datastores(
    datastores,
    root_directory,
    merge_recordings=True,
    merge_analysis=True,
    merge_stimuli=True,
    replace=False,
):
    """
    This function takes a tuple of datastore in input and merge them into one single datastore which will be saved in root_directory.
    The type of data that should be merged can be controlled through the merge_recordings, merge_analysis and merge_stimuli booleans
    It returns this datastore as a Datastore object.
    """
    merged_datastore = PickledDataStore(
        load=False,
        parameters=ParameterSet(
            {"root_directory": root_directory, "store_stimuli": merge_stimuli}
        ),
        replace=replace,
    )
    j = 0

    # Here we check if sheets and neurons are the same in all datastores
    assert compare_sheets_datastores(
        datastores
    ), "All datastores should contain the same sheets"
    assert compare_neurons_ids_datastores(
        datastores
    ), "Neurons in the datastores should have the same ids"
    assert compare_neurons_position_datastores(
        datastores
    ), "Neurons in the datastores should have the same position"
    assert compare_neurons_annotations_datastores(
        datastores
    ), "Neurons in the datastores should have the same annotations"

    if not os.path.isdir(root_directory):
        os.makedirs(root_directory)

    # Change the block annotations so that it gets the merged version of the experiment parameters
    merged_datastore.block.annotations = datastores[0].block.annotations
    merged_datastore.block.annotations[
        "experiment_parameters"
    ] = merge_experiment_parameters_datastores(datastores)

    j = 0
    for datastore in datastores:

        # Merge the recording of all the datastores if this flag is set to true
        if merge_recordings:
            segments = datastore.get_segments()
            segments += datastore.get_segments(null=True)
            for seg in segments:
                for s in merged_datastore.get_segments():
                    if seg.annotations == s.annotations and seg.null == s.null:
                        print(
                            "Warning: A segment with the same parametrization was already added in the datastore.: %s"
                            % (seg.annotations)
                        )
                        raise ValueError(
                            "A segment with the same parametrization was already added in the datastore already added in the datastore. Currently uniqueness is required. User should check what caused this and modify his simulations  to avoid this!: %s \n %s"
                            % (str(seg.annotations), str(s.annotations))
                        )

                # Load the full segment and adds it to the merged datastore
                if not seg.full:
                    seg.load_full()
                merged_datastore.block.segments.append(
                    PickledDataStoreNeoWrapper(
                        seg, "Segment" + str(j), root_directory, null=seg.null
                    )
                )
                merged_datastore.stimulus_dict[seg.annotations["stimulus"]] = True

                # Create a new pickle file for this mozaik segment and store a corresponding neo segment there
                f = open(root_directory + "/" + "Segment" + str(j) + ".pickle", "wb")
                s = Segment(
                    description=seg.description,
                    file_origin=seg.file_origin,
                    file_datetime=seg.file_datetime,
                    rec_datetime=seg.rec_datetime,
                    index=seg.index,
                    **seg.annotations
                )
                s.spiketrains = seg.spiketrains
                s.analogsignals = seg.analogsignals
                pickle.dump(s, f)

                # Release each segment once it has been added to the merged datastore to save memory
                seg.release()
                j = j + 1

        # Merge the analysis of all the datastores if this flag is set to true
        if merge_analysis:
            adss = datastore.get_analysis_result()
            for ads in adss:
                merged_datastore.add_analysis_result(ads)

        # Merge the stimuli all the datastores if this flag is set to true
        if merge_stimuli:
            for key, value in datastore.sensory_stimulus.items():
                merged_datastore.sensory_stimulus[key] = value

    return merged_datastore
