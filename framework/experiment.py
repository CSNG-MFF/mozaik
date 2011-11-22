from MozaikLite.stimuli.topographica_based import FullfieldDriftingSinusoidalGrating, Null, NaturalImageWithEyeMovement
from MozaikLite.analysis.analysis import AveragedOrientationTuning,  GSTA, Precision
from MozaikLite.visualization.plotting import GSynPlot,RasterPlot,VmPlot,CyclicTuningCurvePlot,OverviewPlot, ConductanceSignalListPlot, RetinalInputMovie
from MozaikLite.visualization.jens_paper_plots import Figure2
from NeuroTools.parameters import ParameterSet, ParameterDist
from MozaikLite.storage.queries import TagBasedQuery, select_result_sheet_query

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
            print 'Presenting stimulus: ',str(s) , '\n'
            (segments,retinal_input) = self.model.present_stimulus_and_record(s)
            data_store.add_recording(segments,s)
            data_store.add_retinal_stimulus(retinal_input,s)
    
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
                                ],ParameterSet({})))    

    def do_analysis(self,data_store):
        print 'Doing Analysis'
        AveragedOrientationTuning(data_store,ParameterSet({})).analyse()
        GSTA(data_store,ParameterSet({'neurons' : [0], 'length' : 50.0 }),tags=['GSTA1']).analyse()
        Precision(select_result_sheet_query(data_store,"V1_Exc"),ParameterSet({'neurons' : [0], 'bin_length' : 1.0 })).analyse()
        
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Inh'})).plot()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'X_ON'})).plot()
        Figure2(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        RetinalInputMovie(data_store,ParameterSet({'frame_rate': 10})).plot()
        
class MeasureNaturalImagesWithEyeMovement(Experiment):
    
    def __init__(self,model,stimulus_duration,num_trials):
        self.model = model
        for k in xrange(0,num_trials):
            self.stimuli.append(NaturalImageWithEyeMovement([   
                            7, # frame duration (roughly like a movie) - is this fast enough?
                            model.visual_field.size[0], 
                            model.visual_field.size[0], 
                            0.0,
                            0.0,
                            90.0, #max_luminance 
                            stimulus_duration, # stimulus duration
                            2, #density
                            k, # trial number
                            40, # x size of image
                            6.66, # eye movement period
                            1 # idd
                        ],'eye_path.pickle','image_naturelle_HIGH.bmp'))    


    def do_analysis(self,data_store):
        print 'Doing Analysis'
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Inh'})).plot()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'X_ON'})).plot()
        RetinalInputMovie(data_store,ParameterSet({'frame_rate': 10})).plot()

class MeasureSpontaneousActivity(Experiment):
    
    def __init__(self,model,duration):
            self.stimuli.append(Null([   
                            7, # frame duration (roughly like a movie) - is this fast enough?
                            model.visual_field.size[0], 
                            model.visual_field.size[0], 
                            0.0,
                            0.0,
                            90.0, #max_luminance 
                            duration, # stimulus duration
                            40, #density
                            0 # trial number
                        ],ParameterSet({})))    

    def do_analysis(self,data_store):
        print 'Doing Analysis'
        #OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        #OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Inh'})).plot()
