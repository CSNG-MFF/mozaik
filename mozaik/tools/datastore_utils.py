import mozaik
from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet
import numpy
import re
logger = mozaik.getMozaikLogger()

def compare_sheets_datastores(datastores):
    sheets = datastores[0].block.annotations['neuron_ids'].keys()
    for i in range(1, len(datastores)):
        if not numpy.array_equal(datastores[i].block.annotations['neuron_ids'].keys(), sheets):
            return False
    return True


def compare_neurons_ids_datastores(datastores):
    sheets = datastores[0].block.annotations['neuron_ids'].keys()
    for sheet in sheets:
        ids = datastores[0].block.annotations['neuron_ids'][sheet]
        for i in range(1, len(datastores)):
            if not numpy.array_equal(datastores[i].block.annotations['neuron_ids'][sheet], ids):
                return False
    return True

def compare_neurons_position_datastores(datastores):
    sheets = datastores[0].block.annotations['neuron_positions'].keys()
    for sheet in sheets:
        positions = datastores[0].block.annotations['neuron_positions'][sheet]
        for i in range(1, len(datastores)):
            if not numpy.array_equal(datastores[i].block.annotations['neuron_positions'][sheet], positions):
                return False
    return True

def compare_neurons_annotations_datastores(datastores):
    sheets = datastores[0].block.annotations['neuron_annotations'].keys()
    for sheet in sheets:
        annotations = datastores[0].block.annotations['neuron_annotations'][sheet]
        for i in range(1, len(datastores)):
            if not numpy.array_equal(datastores[i].block.annotations['neuron_annotations'][sheet], annotations):
                return False
    return True

def merge_experiment_parameters_datastores(datastores):
    experiment_parameters = []
    for datastore in datastores:
        for parameters in datastore.block.annotations['experiment_parameters']:
            experiment_parameters.append(parameters)
    return experiment_parameters

def merge_datastores(datastores, root_directory, merge_recordings = True, merge_analysis = True, merge_stimuli = True, replace = False):
    """
    This function takes a tuple of datastore in input and merge them into one single datastore which will be saved in root_directory.
    The type of data that should be merged can be controlled through the merge_recordings, merge_analysis and merge_stimuli booleans
    It returns this datastore as a Datastore object.
    """
    merged_datastore = PickledDataStore(load=False, parameters=ParameterSet({'root_directory': root_directory,'store_stimuli' : merge_stimuli}), replace = replace)
    j = 0
    assert compare_sheets_datastores(datastores), "All datastores should contain the same sheets"
    assert compare_neurons_ids_datastores(datastores), "Neurons in the datastores should have the same ids"
    assert compare_neurons_position_datastores(datastores), "Neurons in the datastores should have the same position"
    assert compare_neurons_annotations_datastores(datastores), "Neurons in the datastores should have the same annotations"

    merged_datastore.block.annotations = datastores[0].block.annotations
    merged_datastore.block.annotations['experiment_parameters'] = merge_experiment_parameters_datastores(datastores)


    for datastore in datastores:
        if merge_recordings:
            segments = datastore.get_segments()
            for seg in segments:
                seg.identifier = 'Segment'+str(j)
                for s in merged_datastore.get_segments():
                    if seg.annotations == s.annotations:
                        print("Warning: A segment with the same parametrization was already added in the datastore.: %s" % (seg.annotations))
                        raise ValueError("A segment with the same parametrization was already added in the datastore already added in the datastore. Currently uniqueness is required. User should check what caused this and modify his simulations  to avoid this!: %s \n %s" % (str(seg.annotations),str(s.annotations)))

                merged_datastore.block.segments.append(seg)
                merged_datastore.stimulus_dict[seg.annotations['stimulus']] = True
                j = j + 1

        if merge_analysis:
            adss = datastore.get_analysis_result()
            for ads in adss:
                merged_datastore.add_analysis_result(ads)


        if merge_stimuli:
            for key, value in datastore.sensory_stimulus.items():
                merged_datastore.sensory_stimulus[key] = value

    return merged_datastore


