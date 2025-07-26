"""
Module containing the experiment API.
"""
import numpy
import resource
import mozaik
from collections import OrderedDict
from mozaik.stimuli import InternalStimulus
from parameters import ParameterSet
from mozaik.core import ParametrizedObject
from mozaik.tools.distribution_parametrization import ParameterWithUnitsAndPeriod, MozaikExtendedParameterSet

logger = mozaik.getMozaikLogger()


class Experiment(ParametrizedObject):
    """
    The abastract class for an experiment. 
    
    The experiment defines the list of stimuli that it needs to present to the brain.These stimuli presentations have to be independent - e.g. should not
    temporarily depend on each other. Experiment should also specify the analysis of the
    recorded results that it performs. This can be left empty if analysis will
    be done later.
    
    The experiment has to also define the `direct_stimulation` variable which should contain a list of dictionaries one per each stimulus.
    The keys in these dictionaries are sheet names and values are list of :class:`mozail.sheets.direct_stimulator.DirectStimulator` instances,
    that specify what direct stimulations should be applied to given layers during the corresponding stimulus. Layers to which no direct stimulation 
    is applied can stay empty. Also if the direct_stimulation is set to None, empty dictionaries will be automatically passed to the model, 
    indicating no direct stimulation is required.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
          
    Other parameters
    ----------------

    duration : float (ms)
             The duration of single presentation of the stimulus.


    NOTE
    ----
    When creating a new Expriment, user inherits from the Experiment class, and in the constructor fills up the `self.stimuli` array with the list of stimuli
    that the experiment presents to the model. One can also implement the do_analysis method, which should perform the analysis that the experiments requires
    at the end. 
    """
    
    def __init__(self, model,parameters):
        ParametrizedObject.__init__(self, parameters)
        self.model = model
        self.stimuli = []
        self.direct_stimulation = None
    
    def return_stimuli(self):
        """
        This function is called by mozaik to retrieve the list of stimuli the experiment requires to be presented to the model.
        """
        return self.stimuli
        
    def run(self,data_store,stimulus_indexes):
        """
        This function is called to execute the experiment.
        
        Parameters
        ----------
        
        data_store : DataStore
                   The data store into which to store the recorded data.
                   
        stimulus_indexes : list(Stimulus)
                The indexes of stimuli to present to the model.
        
        Returns
        -------
        strsum : int (s)
               The overal simulation time it took to execute the experiment.
                
        Notes
        -----
        The reason why this function gets a list of stimulus index as input is that even though the experiment itself defines the list of stimuli
        to present to the model, some of these might have already been presented. The module `mozaik.controller` filters
        the list of stimuli which to present to prevent repetitions, and lets this function know via the stimuli argument which stimuli to actually present.
        """
        srtsum = 0
        for i in stimulus_indexes:
            s = self.stimuli[i]
            logger.debug('Presenting stimulus: ' + str(s) + '\n')
            if self.direct_stimulation == None:
               ds = OrderedDict()
            else:
               ds = self.direct_stimulation[i]
            (segments,null_segments,input_stimulus,simulator_run_time,model_exploded) = self.model.present_stimulus_and_record(s,ds)
            srtsum += simulator_run_time
            data_store.add_recording(segments,s)
            data_store.add_stimulus(input_stimulus,s)
            data_store.add_direct_stimulation(ds,s)
            
            if null_segments != []:
               data_store.add_null_recording(null_segments,s) 
            
            logger.info('Stimulus %d/%d finished. Memory usage: %iMB' % (i+1,len(stimulus_indexes),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
            if model_exploded:
                return srtsum, model_exploded
        return srtsum, model_exploded
        
    def do_analysis(self):
        raise NotImplementedError
        pass

class PoissonNetworkKick(Experiment):
    """
    This experiment injects Poisson spike trains into the target popullation.
    
    This experiment does not show any stimulus.
    For the duration of the experiment it will stimulate neurons 
    definded by the recording configurations in recording_configuration_list
    in the sheets specified in the sheet_list with Poisson spike train of mean 
    frequency determined by the corresponding values in lambda_list.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
  
    sheet_list : int
               The list of sheets in which to do stimulation

    drive_period : float (ms)
                 The length of the constant drive, after which it will be linearly taken down to 0 at the end of the stimulation.   
                        
    stimulation_configuration : ParameterSet
                              The parameter set for direct stimulation specifing neurons to which the kick will be administered.
                                 
    lambda_list : list
                List of the means of the Poisson spike train to be injected into the neurons specified in stimulation_configuration (one per each sheet).
    
    weight_list : list
                List of spike sizes of the Poisson spike train to be injected into the neurons specified in stimulation_configuration (one per each sheet).
    """
    
    required_parameters = ParameterSet({
            'duration': float,
            'sheet_list' : list,
            'drive_period' : float,
            'stimulation_configuration' : ParameterSet,
            'lambda_list' : list,
            'weight_list' : list,
    })

    
    def __init__(self,model,parameters):
            Experiment.__init__(self, model,parameters)
            from mozaik.sheets.direct_stimulator import Kick

            d  = OrderedDict()
            for i,sheet in enumerate(self.parameters.sheet_list):
                p = MozaikExtendedParameterSet({'exc_firing_rate' : self.parameters.lambda_list[i],
                                                      'exc_weight' : self.parameters.weight_list[i],
                                                      'drive_period' : self.parameters.drive_period,
                                                      'population_selector' : self.parameters.stimulation_configuration})

                d[sheet] = [Kick(model.sheets[sheet],p)]
            
            self.direct_stimulation = [d]
            self.stimuli.append(
                        InternalStimulus(   
                                            frame_duration=self.parameters.duration, 
                                            duration=self.parameters.duration,
                                            trial=0,
                                            direct_stimulation_name='Kick',
                                            direct_stimulation_parameters = p
                                         )
                                )

class InjTest(Experiment):
    required_parameters = ParameterSet({
            'duration': float,
            'current' : float,
            'sheet_list' : list,
            'stimulation_configuration' : ParameterSet,
        })

    
    def __init__(self,model,parameters):
            Experiment.__init__(self, model,parameters)
            from mozaik.sheets.direct_stimulator import Depolarization

            d  = OrderedDict()
            for i,sheet in enumerate(self.parameters.sheet_list):
                p = MozaikExtendedParameterSet({
                                                'current' : self.parameters.current,
                                                'population_selector' : self.parameters.stimulation_configuration})

                d[sheet] = [Depolarization(model.sheets[sheet],p)]
            
            self.direct_stimulation = [d]
            self.stimuli.append(
                        InternalStimulus(   
                                            frame_duration=self.parameters.duration, 
                                            duration=self.parameters.duration,
                                            trial=0,
                                            direct_stimulation_name='Injection',
                                            direct_stimulation_parameters = p
                                         )
                                )

        
class NoStimulation(Experiment):
    """ 
    This is a special experiment that does not show any stimulus for the duration of the experiment. 

    This experiment is universal, in that it is not dependent on what sensory modality/model is used in the
    given simulation. It will ensure that no sensory stimulation will be performed.  
    
    Notes
    -----
    Unlike :class:`.MeasureSpontaneousActivity` this can be used in model with no sensory input sheet.
    """
    required_parameters = ParameterSet({
                                        'duration': float,
                                       })

    def __init__(self,model,parameters):
        Experiment.__init__(self, model,parameters)
        self.stimuli.append(
                        InternalStimulus(   
                                            frame_duration=self.parameters.duration, 
                                            duration=self.parameters.duration,
                                            trial=0,
                                         )
                                )


