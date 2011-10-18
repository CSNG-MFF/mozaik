from MozaikLite.stimuli.stimulus_generator import FullfieldDriftingSinusoidalGrating, Null
from MozaikLite.analysis.analysis import AveragedOrientationTuning, Neurotools, GSTA
from MozaikLite.visualization.plotting import GSynPlot,RasterPlot,VmPlot,CyclicTuningCurvePlot,OverviewPlot, ConductanceSignalListPlot
from MozaikLite.visualization.jens_paper_plots import Figure2
from NeuroTools.parameters import ParameterSet, ParameterDist
from MozaikLite.storage.queries import TagBasedQuery

import numpy

class Experiment(object):
    """
    The experiment defines the list of stimuli that it needs to present to the brain.
    This stimuli presentations have to be independent - e.g. should not temporarily 
    depend on others. It should also specify the analysis of the recorded results 
    that it performs. This can be left empty if analysis will be done later.
    """
    
    stimuli = []
    
    def return_stimuli(self):
        return self.stimuli
        
    def run(self,data_store,stimuli):
        for s in stimuli:
            print 'Presenting stimulus: ',str(s)
            segments = self.model.present_stimulus_and_record(s)
            data_store.add_recording(segments,s)
    
    def do_analysis(self):
        raise NotImplementedError
        pass

class MeasureOrientationTuningFullfield(Experiment):
    
    def __init__(self,model,num_orientations,spatial_frequency,temporal_frequency,grating_duration,num_trials):
        self.model = model
        for j in [1.0]:
            for i in xrange(0,num_orientations):
                for k in xrange(0,num_trials):
                    self.stimuli.append(FullfieldDriftingSinusoidalGrating([   
                                    7, # frame duration (roughly like a movie) - is this fast enough?
                                    model.visual_field.size[0], 
                                    0.0,
                                    0.0,
                                    j*90.0, #max_luminance 
                                    grating_duration, # stimulus duration
                                    40, #density
                                    k, # trial number
                                    numpy.pi/num_orientations*i, #orientation
                                    spatial_frequency,
                                    temporal_frequency, #stimulus duration - we want to get one full sweep of phases
                                ]))    

    def do_analysis(self,data_store):
        print 'Doing Analysis'
        AveragedOrientationTuning(data_store,ParameterSet({})).analyse()
        GSTA(data_store,ParameterSet({'neurons' : [0], 'length' : 50.0 }),tags=['GSTA1']).analyse()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Inh'})).plot()
        Figure2(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        


class MeasureSpontaneousActivity(Experiment):
    
    def __init__(self,model,duration):
            self.stimuli.append(Null([   
                            7, # frame duration (roughly like a movie) - is this fast enough?
                            model.visual_field.size[0], 
                            0.0,
                            0.0,
                            90.0, #max_luminance 
                            duration, # stimulus duration
                            40, #density
                            0 # trial number
                        ]))    

    def do_analysis(self,data_store):
        print 'Doing Analysis'
        Neurotools(data_store).analyse()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Inh'})).plot()
