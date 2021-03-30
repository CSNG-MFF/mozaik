from mozaik.storage import datastore

def merge_datastores(datastores, root_directory, merge_recordings = True, merge_analysis = True, merge_stimuli = True):
    """
    This function takes a tuple of datastore in input and merge them into one single datastore which will be saved in root_directory.
    The type of data that should be merged can be controlled through the merge_recordings, merge_analysis and merge_stimuli booleans
    It returns this datastore as a Datastore object.
    """
    merged_datastore = PickledDataStore(load=False, parameters=MozaikExtendedParameterSet({'root_directory': root_directory,'store_stimuli' : merge_stimuli}, replace = True))

    for datastore in datastores:
       
        if merge_recordings:
            segments = datastore.get_segments()
            for seg in segments:
                if merged_datastore.stimulus_dict[seg.annotations['stimulus']]: 
                    logger.info("Warning: A segment corresponding to the same stimulus was already added in the datastore.: %s" % (str(result)))
                    for i,s in enumerate(self.block.segments):
                        if merged_datastore.stimulus_dict[s.annotations['stimulus']]:
                            merged_datastore.block.segments[i] = s
                            break
                else:
                    merged_datastore.block.segments.append(s)
                    merged_datastore.stimulus_dict[str(stimulus)] = True
                
        if merge_analysis:
            adss = datastore.get_analysis_results()
            for ads in adss:
                merged_datastore.add_analysis_results(ads) 


        if merge_stimuli:
            for key, value in datastore.sensory_stimulus:
                merged_datastore.sensory_stimulus[key] = value

    return merged_datastore 
