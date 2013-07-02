"""
Module containing the experiment API.
"""
import mozaik
from mozaik.stimuli import InternalStimulus
import numpy
import resource

logger = mozaik.getMozaikLogger()


class Experiment(object):
    """
    The abastract class for an experiment. The experiment defines the list of 
    stimuli that it needs to present to the brain.These stimuli presentations have to be independent - e.g. should not
    temporarily depend on each other. Experiment should also specify the analysis of the
    recorded results that it performs. This can be left empty if analysis will
    be done later.
    
    Also each Experiment can define the spike_stimulator and current_stimulator variables that 
    allow the experiment to stimulate specific neurons in the model with either spike trains and conductances.
    
    The (exc/inh)_spike_stimulator is a dictionary where keys are the name of the sheets, and the values are tuples (neuron_list,spike_generator)
    where neuron_list is a list of neuron_ids that should be stimulated, this varialbe can also be a string 'all', in which case all neurons 
    in the given sheet will be stimulated. The spike_generator should be function that receives a single input - duration - that returns a 
    spike train (list of spike times) of lasting the duration miliseconds. This function will be called for each stimulus presented during this experiment,
    with the duration of the stimulus as the duration parameter. 
    
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    NOTE
    ----
    When creating a new Expriment, user inherits from the Experiment class, and in the constructor fills up the `self.stimuli` array with the list of stimuli
    that the experiment presents to the model. One can also implement the do_analysis method, which should perform the analysis that the experiments requires
    at the end. Finaly the `self.exc_spike_stimulators` and `self.inh_spike_stimulators` dictionaries can also be specified in the constructor.
    
    The spikes from (exc/inh)_spike_stimulator will be weighted by the connection with weights defined by the sheet's background_noise.(exc/inh)_weight parameter!
    """
    def __init__(self, model):
        self.model = model
        self.stimuli = []
        self.exc_spike_stimulators = {}
        self.inh_spike_stimulators = {}
        self.current_stimulators = {}
    
    def return_stimuli(self):
        """
        This function is called by mozaik to retrieve the list of stimuli the experiment requires to be presented to the model.
        """
        return self.stimuli
        
    def run(self,data_store,stimuli):
        """
        This function is called to execute the experiment.
        
        Parameters
        ----------
        
        data_store : DataStore
                   The data store into which to store the recorded data.
                   
        stimuli : list(Stimulus)
                The list of stimuli to present to the model.
        
        Returns
        -------
        strsum : int (s)
               The overal simulation time it took to execute the experiment.
                
        Notes
        -----
        The reason why this function gets a list of stimuli as input is that even though the experiment itself defines the list of stimuli
        to present to the model, some of these might have already been presented. The module `mozaik.controller` filters
        the list of stimuli which to present to prevent repetitions, and lets this function know via the stimuli argument which stimuli to actually present.
        """
        srtsum = 0
        for i,s in enumerate(stimuli):
            logger.debug('Presenting stimulus: ' + str(s) + '\n')
            (segments,input_stimulus,simulator_run_time) = self.model.present_stimulus_and_record(s,self.exc_spike_stimulators,self.inh_spike_stimulators)
            srtsum += simulator_run_time
            data_store.add_recording(segments,s)
            data_store.add_stimulus(input_stimulus,s)
            logger.info('Stimulus %d/%d finished. Memory usage: %iMB' % (i+1,len(stimuli),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
        return srtsum
        
    def do_analysis(self):
        raise NotImplementedError
        pass

class PoissonNetworkKick(Experiment):
    """
    This experiment does not show any stimulus.
    Importantly for the duration of the experiment it will stimulate neurons 
    definded by the recording configurations in recording_configuration_list
    in the sheets specified in the sheet_list with Poisson spike train of mean 
    frequency determined by the corresponding values in lambda_list.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.


    duration : str
             The duration of single presentation of the stimulus.
    
    sheet_list : int
               The list of sheets in which to do stimulation
               
    recording_configuration_list : list
                                 The list of recording configurations (one per each sheet).
                                 
    lambda_list : list
                List of the means of the Poisson spike train to be injected into the neurons specified in recording_configuration_list (one per each sheet).
    """
    def __init__(self,model,duration,sheet_list,recording_configuration_list,lambda_list):
            Experiment.__init__(self, model)
            from NeuroTools import stgen
            for sheet_name,lamb,rc in zip(sheet_list,lambda_list,recording_configuration_list):
                idlist = rc.generate_idd_list_of_neurons()
                seeds=mozaik.get_seeds((len(idlist),))
                stgens = [stgen.StGen(seed=seeds[i]) for i in xrange(0,len(idlist))]
                generator_functions = [(lambda duration,lamb=lamb,stgen=stgens[i]: stgen.poisson_generator(rate=lamb,t_start=0,t_stop=duration).spike_times) for i in xrange(0,len(idlist))]
                self.exc_spike_stimulators[sheet_name] = (list(idlist),generator_functions)

            self.stimuli.append(
                        InternalStimulus(   
                                            frame_duration=duration, 
                                            duration=duration,
                                            trial=0,
                                            direct_stimulation_name='Kick'
                                         )
                                )
        
class NoStimulation(Experiment):
    """
    This experiment does not show any stimulus for the duration of the experiment.
    
    Notes
    -----
    Unlike :class:`.MeasureSpontaneousActivity` this can be used in model with no sensory input sheet.
    """
    
    def __init__(self,model,duration):
        Experiment.__init__(self, model)
        self.stimuli.append(
                        InternalStimulus(   
                                            frame_duration=duration, 
                                            duration=duration,
                                            trial=0,
                                            direct_stimulation_name='None'
                                         )
                                )
