from MozaikLite.stimuli.stimulus_generator import FullfieldDriftingSinusoidalGrating, Null
from MozaikLite.analysis.analysis import AveragedOrientationTuning, Neurotools, GSTA
from MozaikLite.visualization.plotting import GSynPlot,RasterPlot,VmPlot,CyclicTuningCurvePlot,OverviewPlot, ConductanceSignalListPlot
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
        
    def run(self,model,data_store,stimuli):
        for s in stimuli:
            print 'Presenting stimulus: ',str(s)
            segment = model.present_stimulus_and_record(s)
            data_store.add_recording(segment,s)
    
    def do_analysis(self):
        pass

class MeasureOrientationTuningFullfield(Experiment):
    
    def __init__(self,model,num_orientations,spatial_frequency,temporal_frequency,grating_duration):
        
        for j in [1.0]:
            for i in xrange(0,num_orientations):
                self.stimuli.append(FullfieldDriftingSinusoidalGrating([   
                                7, # frame duration (roughly like a movie) - is this fast enough?
                                model.visual_field.size[0], 
                                0.0,
                                0.0,
                                j*90.0, #max_luminance 
                                grating_duration, # stimulus duration
                                40, #density
                                numpy.pi/num_orientations*i, #orientation
                                spatial_frequency,
                                temporal_frequency, #stimulus duration - we want to get one full sweep of phases
                            ]))    

    def do_analysis(self,data_store):
        print 'Doing Analysis'
        AveragedOrientationTuning(data_store,ParameterSet({})).analyse()
        Neurotools(data_store,ParameterSet({})).analyse()
        GSTA(data_store,ParameterSet({'neurons' : [0], 'length' : 10.0 }),tags=['GSTA1']).analyse()
            
        ConductanceSignalListPlot( TagBasedQuery(ParameterSet({'tags' : ['GSTA1'] })).query(data_store), ParameterSet({'sheet_name' : 'V1_Exc'}) ).plot()
            
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Inh'})).plot()
        CyclicTuningCurvePlot(data_store,ParameterSet({'neuron' : 0, 'tuning_curve_name' : 'TuningCurve', 'ylabel' : 'Activity', 'sheet_name' : 'V1_Exc'})).plot()
        CyclicTuningCurvePlot(data_store,ParameterSet({'neuron' : 0, 'tuning_curve_name' : 'TuningCurve', 'ylabel' : 'Activity', 'sheet_name' : 'V1_Inh'})).plot()


class MeasureSpontaneousActivity(Experiment):
    
    def __init__(self,model,duration):
            self.stimuli.append(Null([   
                            7, # frame duration (roughly like a movie) - is this fast enough?
                            model.visual_field.size[0], 
                            0.0,
                            0.0,
                            90.0, #max_luminance 
                            duration, # stimulus duration
                            40 #density
                        ]))    

    def do_analysis(self,data_store):
        print 'Doing Analysis'
        Neurotools(data_store).analyse()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Exc'})).plot()
        OverviewPlot(data_store,ParameterSet({'sheet_name' : 'V1_Inh'})).plot()
