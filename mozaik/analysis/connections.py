import numpy
import quantities as qt
from .analysis import Analysis
from mozaik.analysis.data_structures import SingleValue
from mozaik.analysis.data_structures import PerNeuronValue
from parameters import ParameterSet
from mozaik.storage import queries
import mozaik

from builtins import zip

logger = mozaik.getMozaikLogger()

class InputAnalysis(Analysis):
    """
    """
    required_parameters = ParameterSet({
        'afferent_connections': list,  # The name of the parameters through which to average when calculating the averaged firing rate
        'recurrent_connections': list,  # The name of the parameters through which to average when calculating the averaged firing rate
        'inhibitory_connections': list,  # The name of the parameters through which to average when calculating the averaged firing rate
        'feedback_connections': list,  # The name of the parameters through which to average when calculating the averaged firing rate
        'sheet_name': str,
        'neuron_ids': list,
        'local_connections_range': float,
    })

    def perform_analysis(self):

        connections = self.parameters.afferent_connections + self.parameters.recurrent_connections + self.parameters.feedback_connections
        aff_dic = {}
        loc_dic = {}
        dist_dic = {}
        inh_dic = {}
        fb_dic = {}

        for connection in connections:
            conn = self.datastore.get_analysis_result(identifier='Connections', target_name=self.parameters.sheet_name, proj_name=connection)[0]
            adss = queries.param_filter_query(self.datastore, sheet_name=conn.source_name, value_name='Firing rate', identifier='PerNeuronValue').get_analysis_result() 
            for ads in adss:
                st = ads.stimulus_id
                aff_inp = []
                sheet_idx = self.datastore.get_sheet_indexes(self.parameters.sheet_name,self.parameters.neuron_ids)
                for i in range(len(sheet_idx)):

                    presyn_idx = numpy.nonzero(conn.weights[:,1].flatten()==sheet_idx[i])[0] 
                    presyn_id_sheet = conn.weights[presyn_idx,0].astype(int) 

                    if connection in self.parameters.recurrent_connections:
                        x = self.datastore.get_neuron_positions()[self.parameters.sheet_name][0][sheet_idx[i]]
                        y = self.datastore.get_neuron_positions()[self.parameters.sheet_name][1][sheet_idx[i]]
                        presyn_x = self.datastore.get_neuron_positions()[conn.source_name][0][presyn_id_sheet]
                        presyn_y = self.datastore.get_neuron_positions()[conn.source_name][1][presyn_id_sheet]
                        x_diff = presyn_x - x
                        y_diff = presyn_y - y
                        local_ids = presyn_id_sheet[numpy.nonzero(numpy.sqrt(numpy.multiply(x_diff, x_diff)+numpy.multiply(y_diff, y_diff)) < self.parameters.local_connections_range)[0]]
                        distal_ids = presyn_id_sheet[numpy.nonzero(numpy.sqrt(numpy.multiply(x_diff, x_diff)+numpy.multiply(y_diff, y_diff)) > self.parameters.local_connections_range)[0]]
                            
                        frs_loc = ads.get_value_by_id(self.datastore.get_sheet_ids(conn.source_name,local_ids))
                        frs_dist = ads.get_value_by_id(self.datastore.get_sheet_ids(conn.source_name,distal_ids))
                        weights_loc = conn.weights[local_ids, 2]
                        weights_dist = conn.weights[distal_ids, 2]
                        inp_loc = sum([frs_loc[j] * weights_loc[j] for j in range(len(local_ids))]) 
                        inp_dist = sum([frs_dist[j] * weights_dist[j] for j in range(len(distal_ids))]) 
                        
                        if st in loc_dic.keys():
                            loc_dic[st][i] += inp_loc
                        else:
                            loc_dic[st] = [0] * len(self.parameters.neuron_ids)
                            loc_dic[st][i] += inp_loc

                        if st in dist_dic.keys():
                            dist_dic[st][i] += inp_dist
                        else:
                            dist_dic[st] = [0] * len(self.parameters.neuron_ids)
                            dist_dic[st][i] += inp_dist
                    else:
                        presyn_ids = self.datastore.get_sheet_ids(conn.source_name,conn.weights[presyn_id_sheet, 0].astype(int)) 
                        frs = ads.get_value_by_id(presyn_ids)
                        weights = conn.weights[presyn_id_sheet, 2]
                        inp = sum([frs[j] * weights[j] for j in range(len(presyn_ids))]) 
                        if connection in self.parameters.afferent_connections:
                            if st in aff_dic.keys():
                                aff_dic[st][i] += inp
                            else:
                                aff_dic[st] = [0] * len(self.parameters.neuron_ids)
                                aff_dic[st][i] += inp

                        elif connection in self.parameters.inhibitory_connections:
                            if st in inh_dic.keys():
                                inh_dic[st][i] += inp
                            else:
                                inh_dic[st] = [0] * len(self.parameters.neuron_ids)
                                inh_dic[st][i] += inp

                        elif connection in self.parameters.feedback_connections:
                            if st in fb_dic.keys():
                                fb_dic[st][i] += inp
                            else:
                                fb_dic[st] = [0] * len(self.parameters.neuron_ids)
                                fb_dic[st][i] += inp


        for st, aff_inp in aff_dic.items():  
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(aff_inp,self.parameters.neuron_ids,qt.Dimensionless,
                                   stimulus_id=st,
                                   value_name='Afferent Input',
                                   sheet_name=self.parameters.sheet_name,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))

        for st, loc_inp in loc_dic.items():
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(loc_inp,self.parameters.neuron_ids,qt.Dimensionless,
                                   stimulus_id=st,
                                   value_name='Local Input',
                                   sheet_name=self.parameters.sheet_name,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))

        for st, dist_inp in dist_dic.items():
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(dist_inp,self.parameters.neuron_ids,qt.Dimensionless,
                                   stimulus_id=st,
                                   value_name='Long-range Input',
                                   sheet_name=self.parameters.sheet_name,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))


        for st, inh_inp in inh_dic.items():
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(inh_inp,self.parameters.neuron_ids,qt.Dimensionless,
                                   stimulus_id=st,
                                   value_name='Inhibitory Input',
                                   sheet_name=self.parameters.sheet_name,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))

        for st, fb_inp in fb_dic.items(): 
                self.datastore.full_datastore.add_analysis_result(
                    PerNeuronValue(fb_inp,self.parameters.neuron_ids,qt.Dimensionless,
                                   stimulus_id=st,
                                   value_name='Feedback Input',
                                   sheet_name=self.parameters.sheet_name,
                                   tags=self.tags,
                                   analysis_algorithm=self.__class__.__name__,
                                   period=None))

