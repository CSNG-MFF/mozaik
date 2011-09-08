from MozaikLite.stimuli.stimulus_generator import FullfieldDriftingSinusoidalGrating, Null
from MozaikLite.analysis.analysis import RasterPlot,OrientationTuning, VmPlot, GSynPlot
import numpy

class Experiment(object):
    """
    The experiment defines the list of stimuli that it needs to present to the brain.
    This stimuli presentations have to be independent - e.g. should not temporarily 
    depend on others.
    """
    
    stimuli = []
    
    def return_stimuli(self):
        return self.stimuli
        
    def run(self,model,data_store,stimuli):
        for s in stimuli:
            print 'Presenting stimulus: ',str(s)
            segment = model.present_stimulus_and_record(s)
            data_store.add_recording(segment,s)

class MeasureOrientationTuningFullfield(Experiment):
    
    def __init__(self,model,num_orientations,spatial_frequency,temporal_frequency,grating_duration):
        for i in xrange(0,num_orientations):
            self.stimuli.append(FullfieldDriftingSinusoidalGrating([   
                            7, # frame duration (roughly like a movie) - is this fast enough?
                            model.visual_field.size[0], 
                            0.0,
                            0.0,
                            90.0, #max_luminance 
                            grating_duration, # stimulus duration
                            40, #density
                            numpy.pi/num_orientations*i, #orientation
                            spatial_frequency,
                            temporal_frequency, #stimulus duration - we want to get one full sweep of phases
                        ]))    

    def do_analysis(self,data_store):
        print 'Doing Analysis'
        #RasterPlot(data_store).analyse()
        VmPlot(data_store).analyse()
        GSynPlot(data_store).analyse()
        OrientationTuning(data_store).analyse()
        


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
        RasterPlot(data_store).analyse()
        VmPlot(data_store).analyse()
        GSynPlot(data_store).analyse()
        OrientationTuning(data_store).analyse()
        
