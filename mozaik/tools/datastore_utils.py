import mozaik
from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet

logger = mozaik.getMozaikLogger()

def merge_datastores(datastores, root_directory, merge_recordings = True, merge_analysis = True, merge_stimuli = True):
    """
    This function takes a tuple of datastore in input and merge them into one single datastore which will be saved in root_directory.
    The type of data that should be merged can be controlled through the merge_recordings, merge_analysis and merge_stimuli booleans
    It returns this datastore as a Datastore object.
    """
    merged_datastore = PickledDataStore(load=False, parameters=ParameterSet({'root_directory': root_directory,'store_stimuli' : merge_stimuli}), replace = True)
    j = 0
    for datastore in datastores:
        if merge_recordings:
            segments = datastore.get_segments()
            for seg in segments:
                segment_exists = False
                seg.identifier = j
                for i, s in enumerate(merged_datastore.block.segments):
                    if seg.annotations == s.annotations:
                        print("Warning: A segment with the same parametrization was already added in the datastore.: %s" % (seg.annotations['stimulus']))
                        merged_datastore.block.segments[i] = seg
                        segment_exists = True
                        break

                if not segment_exists:
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

