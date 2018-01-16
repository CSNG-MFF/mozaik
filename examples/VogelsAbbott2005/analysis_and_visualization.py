import numpy
import mozaik
from mozaik.visualization.plotting import *
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
from mozaik.analysis.analysis import *
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore


def perform_analysis_and_visualization(data_store):
    analog_ids = sorted(param_filter_query(data_store,sheet_name="Exc_Layer").get_segments()[0].get_stored_esyn_ids())
    analog_ids_inh = sorted(param_filter_query(data_store,sheet_name="Inh_Layer").get_segments()[0].get_stored_esyn_ids())
    spike_ids = sorted(param_filter_query(data_store,sheet_name="Exc_Layer").get_segments()[0].get_stored_spike_train_ids())
    spike_ids_inh = sorted(param_filter_query(data_store,sheet_name="Inh_Layer").get_segments()[0].get_stored_spike_train_ids())
    
    if True: # PLOTTING
            activity_plot_param =    {
                   'frame_rate' : 5,  
                   'bin_width' : 5.0, 
                   'scatter' :  True,
                   'resolution' : 0
            }       
            
            
            PSTH(param_filter_query(data_store,st_direct_stimulation_name="None"),ParameterSet({'bin_length' : 5.0})).analyse()
            TrialAveragedFiringRate(param_filter_query(data_store,st_direct_stimulation_name="None"),ParameterSet({})).analyse()
            Irregularity(param_filter_query(data_store,st_direct_stimulation_name="None"),ParameterSet({})).analyse()
            NeuronToNeuronAnalogSignalCorrelations(param_filter_query(data_store,analysis_algorithm='PSTH'),ParameterSet({'convert_nan_to_zero' : True})).analyse()
            PopulationMeanAndVar(data_store,ParameterSet({})).analyse()
            
            data_store.print_content(full_ADS=True)
            
            dsv = param_filter_query(data_store,st_direct_stimulation_name=None)    
            
            OverviewPlot(dsv,ParameterSet({'sheet_name' : 'Exc_Layer', 'neuron' : analog_ids[0], 'sheet_activity' : {}, 'spontaneous' : False}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name='ExcAnalog1.png').plot({'Vm_plot.y_lim' : (-80,-50),'Conductance_plot.y_lim' : (0,500.0)})
            OverviewPlot(dsv,ParameterSet({'sheet_name' : 'Exc_Layer', 'neuron' : analog_ids[1], 'sheet_activity' : {}, 'spontaneous' : False}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name='ExcAnalog2.png').plot({'Vm_plot.y_lim' : (-80,-50),'Conductance_plot.y_lim' : (0,500.0)})    
            OverviewPlot(dsv,ParameterSet({'sheet_name' : 'Exc_Layer', 'neuron' : analog_ids[2], 'sheet_activity' : {}, 'spontaneous' : False}),fig_param={'dpi' : 100,'figsize': (19,12)},plot_file_name='ExcAnalog3.png').plot({'Vm_plot.y_lim' : (-80,-50),'Conductance_plot.y_lim' : (0,500.0)})
            
            
            RasterPlot(dsv,ParameterSet({'sheet_name' : 'Exc_Layer', 'neurons' : spike_ids,'trial_averaged_histogram': False, 'spontaneous': False}),fig_param={'dpi' : 100,'figsize': (17,5)},plot_file_name='ExcRaster.png').plot({'SpikeRasterPlot.group_trials':True})
            RasterPlot(dsv,ParameterSet({'sheet_name' : 'Inh_Layer', 'neurons' : spike_ids_inh,'trial_averaged_histogram': False, 'spontaneous': False}),fig_param={'dpi' : 100,'figsize': (17,5)},plot_file_name='InhRaster.png').plot({'SpikeRasterPlot.group_trials':True})

