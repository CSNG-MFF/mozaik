# encoding: utf-8
import mozaik
import numpy
import ast
from mozaik.connectors import Connector
from mozaik.connectors.modular_connector_functions import *
from collections import Counter
from parameters import ParameterSet
from mozaik.tools.misc import sample_from_bin_distribution, normal_function
from mozaik import load_component
from mozaik.tools.distribution_parametrization import PyNNDistribution
from collections import OrderedDict


from builtins import zip

logger = mozaik.getMozaikLogger()


class ExpVisitor(ast.NodeVisitor):
    """
    AST tree visitor used for determining list of variables in the delay or weight expresions
    """
    
    def __init__(self,**params):
        ast.NodeVisitor.__init__(self,**params)
        self.names = []
    
    def visit_Name(self, node):
        node.id
        if not (node.id in self.names):
           self.names.append(node.id) 

class ModularConnector(Connector):
    """
    An abstract connector than allows for mixing of various factors that can affect the connectivity.
    
    The connector sepparates the implementation of delays from the implementation of weights.
    
    It receives a dictionary of weight functions and a dictonary of delays functions each being an instance of ModularConnectorFunction. 
    In both cases the list of functions is combined by using expression string which is a parameter of this class (see parameters for details). 
    
    The values returned by the ModularConnectorFunction will be considered to be in miliseconds when used for specifying delays, or the units used by pyNN for weights 
    in case of specifying weights.
    
    The ModularConnector then sets such computed values of weights and delays directly in the connections.
    
    
    """

    required_parameters = ParameterSet({
        'weight_functions' : ParameterSet, # a dictionary of ModularConnectorFunction's and their parameters that will be used to determine the weights.
                                           # strucutured as follows
                                           #            {
                                           #                 component : 'class_name_of_the_ModularConnectorFunction',
                                           #                 params : {
                                           #                           ...
                                           #                         }
                                           #             }
        'delay_functions' : ParameterSet, # the same as weight_functions but for delays
        'weight_expression' : str, # a python expression that can use variables f1..fn where n is the number of functions in weight_functions, and fi corresponds to the name given to a ModularConnectorFunction in weight_function ParameterSet. It determines how are the weight functions combined to obtain the weights
        'delay_expression' : str, # a python expression that can use variables f1..fn where n is the number of functions in delays_functions, and fi corresponds to the name given to a ModularConnectorFunction in delays_function ParameterSet. It determines how are the delays functions combined to obtain the delays
        'self_connections' : bool, # whether to allow self connections or not 
    })
    
    def __init__(self, network, name,source, target, parameters):
      Connector.__init__(self, network, name, source,target,parameters)
      
      # lets load up the weight ModularConnectorFunction's
      self.weight_functions = OrderedDict()
      self.delay_functions = OrderedDict()
      self.simulator_time_step = self.sim.get_time_step()
      # lets determine the list of variables in weight expressions
      v = ExpVisitor()
      v.visit(ast.parse(self.parameters.weight_expression))
      self.weight_function_names = v.names
      # lets determine the list of variables in delay expressions
      v = ExpVisitor()
      v.visit(ast.parse(self.parameters.delay_expression))
      self.delay_function_names = v.names
      
      for k in self.weight_function_names:
          self.weight_functions[k] = load_component(self.parameters.weight_functions[k].component)(self.source,self.target,self.parameters.weight_functions[k].params)
          assert isinstance(self.weight_functions[k],ModularConnectorFunction)
          
      for k in self.delay_function_names:
          self.delay_functions[k] = load_component(self.parameters.delay_functions[k].component)(self.source,self.target,self.parameters.delay_functions[k].params)

    def _obtain_weights(self,i,seed=None):
        """
        This function calculates the combined weights from the ModularConnectorFunction in weight_functions
        """
        evaled = OrderedDict()
       
        for k in self.weight_function_names:
            evaled[k] = self.weight_functions[k].evaluate(i) if not seed else self.weight_functions[k].evaluate(i,seed=seed)
        weights = numpy.zeros((self.source.pop.size,)) + eval(self.parameters.weight_expression,globals(),evaled)
        if not self.parameters.self_connections and self.target.name == self.source.name:
            weights[i] = 0
        return weights
        
    def _obtain_delays(self,i,seed=None):
        """
        This function calculates the combined weights from the ModularConnectorFunction in weight_functions
        """
        evaled = OrderedDict()
        for k in self.delay_function_names:
            evaled[k] = self.delay_functions[k].evaluate(i) if not seed else self.delay_functions[k].evaluate(i,seed=seed)
        
        delays = numpy.zeros((self.source.pop.size,)) + eval(self.parameters.delay_expression,globals(),evaled)
        delays = numpy.rint(delays / self.simulator_time_step) * self.simulator_time_step
        return delays
        
    def _connect(self):
        connection_list = []
        z = numpy.zeros((self.target.pop.size,))
        for i in numpy.nonzero(self.target.pop._mask_local)[0]: 
            connection_list.extend(list(zip(numpy.arange(0,self.source.pop.size,1),z+i,self.weight_scaler*self._obtain_weights(i).flatten(),self._obtain_delays(i).flatten())))
        
        self.method = self.sim.FromListConnector(connection_list)
        self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                self.method,
                                synapse_type=self.init_synaptic_mechanisms(),
                                label=self.name,
                                receptor_type=self.parameters.target_synapses)

class LocalModuleConnector(ModularConnector):
    """
    ModularConnector which includes the possibility to define a local module.
    Local connections projecting from neurons outside the module to neurons inside
    the module are to be deleted and replaced by connections coming from inside the 
    local module.
    The local module is defined as a circle delimited by the local_module.in_radius parameter
    Local connections projecting to the local module are considered to be any connection
    coming from neuron laying inside a circle of radius local_module.out_radius.
    This class serves as an interface for the modular probabilistic connectors if they're
    supposed to implement the local module functionality.
    """

    required_parameters = ParameterSet({
        'local_module' : ParameterSet,     # parameters of the local module which local connections would only emerge from the same local module
                                           # none if no local module
                                           # strucutured as follows
                                           #            {
                                           #                 in_radius: float (mm)
                                           #                 out_radius: float (mm)
                                           #             }
    })

    def __init__(self, network, name,source, target, parameters):
        """
        Set two numpy arrays:
        lm_idx: List of ids of neurons within the local module.
        l_idx: List of ids of neurons outside the local module which might send
               local connections to the local module.
        """
        ModularConnector.__init__(self, network, name, source,target,parameters)
        if self.parameters.local_module:
            src_x = self.source.pop.positions[0]
            src_y = self.source.pop.positions[1]
            dist = numpy.sqrt(numpy.multiply(src_x, src_x) + numpy.multiply(src_y, src_y))
            # Local module neurons lie from the center at less `than local_module.in_radius`
            self.lm_idx = numpy.nonzero(dist < self.parameters.local_module.in_radius)[0]
            # Local connections projected to the local module emerge from neurons which lie
            # from the center at less `than local_module.out_radius`
            self.l_idx = numpy.nonzero(numpy.logical_and(dist > self.parameters.local_module.in_radius, dist < self.parameters.local_module.out_radius))[0]
        else:
            self.lm_idx = None
            self.l_idx = None

    def local_module_weight_updates(self, idx, weights):
        """ 
        For each neuron, if it lies within the local module, delete all local 
        connections coming from outside the local module and update its weights. 

        Parameters:
            idx: int
                The id of the post-synaptic neuron

            weights: numpy.nd_array
                    The weights to update
        """
        x = self.target.pop.positions[0][idx]
        y = self.target.pop.positions[1][idx]
        weight_sum = 0
        # Update the weights only if the post-synaptic neuron lies in the local module
        if numpy.sqrt(x * x + y * y) < self.parameters.local_module.in_radius:
            weight_sum = numpy.sum(weights[self.l_idx])
            lm_weight_sum = numpy.sum(weights[self.lm_idx])
            weight_ratio = (weight_sum + lm_weight_sum)/lm_weight_sum

            # Set all the weights of local connections emerging from outside the local module to 0 
            # So that no such connection is created
            weights[self.l_idx] = 0

            # Update the weights of the connection emerging from the local module
            # So that the sum of the weights of the local connections stays constant
            if weight_sum > 0:
                weights[self.lm_idx] = weights[self.lm_idx] * weight_ratio
        return weights

class VariableNumSamplesConnector(LocalModuleConnector):
    """
    Modular connector for which the number of sample connections varies
    accross neurons according to the num_samples_functions and 
    num_samples_expression provided.
    If num_samples_expression is an empty string, num_samples is fixed
    accross neurons
    """

    required_parameters = ParameterSet({
        'num_samples_functions' : ParameterSet, # a dictionary of ModularNumSamplesConnectorFunction's and their parameters that will be used to determine the number of sample connections.
                                           # strucutured as follows
                                           #            {
                                           #                 component : 'class_name_of_the_ModularNumSamplesConnectorFunction',
                                           #                 params : {
                                           #                           ...
                                           #                         }
                                           #             }
        'num_samples_expression' : str, # a python expression that can use variables f1..fn where n is the number of functions in num_samples_functions, and fi corresponds to the name given to a ModularNumSamplesConnectorFunction in weight_function ParameterSet. It determines the coefficient that will be multiplied with num_samples to obtain the final number of sampled connections
        })

    def __init__(self, network, name,source, target, parameters):
        """
        Set two numpy arrays:
        lm_idx: List of ids of neurons within the local module.
        l_idx: List of ids of neurons outside the local module which might send
               local connections to the local module.
        """
        LocalModuleConnector.__init__(self, network, name, source,target,parameters)
        if self.parameters.num_samples_expression:
            self.num_samples_functions = OrderedDict()
            # lets determine the list of variables in weight expressions
            v = ExpVisitor()
            v.visit(ast.parse(self.parameters.num_samples_expression))
            self.num_samples_function_names = v.names

            for k in self.num_samples_function_names:
                self.num_samples_functions[k] = load_component(self.parameters.num_samples_functions[k].component)(self.target,self.parameters.num_samples_functions[k].params)
                assert isinstance(self.num_samples_functions[k],ModularNumSamplesConnectorFunction)
        else:
            self.num_samples_functions = None

    def _obtain_num_samples(self,i,samples):
        """
        This function calculates the combined weights from the ModularConnectorFunction in weight_functions
        """
        evaled = OrderedDict()

        if self.num_samples_functions:
            for k in self.num_samples_function_names:
                evaled[k] = self.num_samples_functions[k].evaluate(i)
            return round(samples *  eval(self.parameters.num_samples_expression,globals(),evaled))
        else:
            return samples

class ModularSamplingProbabilisticConnector(VariableNumSamplesConnector):
    """
    VariableNumSampleConnector that interprets the weights as proportional probabilities of connectivity
    and for each neuron in connections it samples num_samples 
    (modulated for each neuron according to num_samples_functions)
    of connections that actually get realized according to these weights.
    Each such sample connections will have weight equal to
    base_weight but note that there can be multiple
    connections between a pair of neurons in this sample (in which case the
    weights are set to the multiple of the base weights times the number of
    occurrences in the sample).
    The _connect method of this class uses multiprocessing
    """

    required_parameters = ParameterSet({
        'num_samples': PyNNDistribution,
        'base_weight' : PyNNDistribution,
    })


    def _connect(self):
        # Generates a splitted and of cells indices to be passed to each subprocesses
        seeds = mozaik.get_seeds(len(self.target.pop))[numpy.nonzero(self.target.pop._mask_local)[0]]
        splitted_seeds = numpy.array_split(seeds, int(self.model.num_threads))
        splitted_cell_indices = numpy.array_split(
            numpy.nonzero(self.target.pop._mask_local)[0], int(self.model.num_threads)
        )

        # This code will be ran by each subprocess
        import multiprocessing

        def build_connector(indices, seeds, queue_cl, queue_v):
            cli = []
            vi = 0
            for i in range(len(indices)):
                weights = self._obtain_weights(indices[i],seeds[i])

                # If a local module is defined, update the weights accordingly
                if self.parameters.local_module:
                    weights = self.local_module_weight_updates(indices[i], weights)

                delays = self._obtain_delays(indices[i],seeds[i])

                num_samples = self._obtain_num_samples(indices[i], self.parameters.num_samples.next())

                co = Counter(
                    sample_from_bin_distribution(
                        weights, int(num_samples), seeds[i]
                    )
                )

                vi = vi + numpy.sum(list(co.values()))
                k = list(co.keys())
                a = numpy.array(
                    [
                        k,
                        numpy.zeros(len(k)) + indices[i],
                        self.weight_scaler
                        * numpy.multiply(
                            self.parameters.base_weight.copy(seeds[i]).next(len(k)),
                            list(co.values())
                        ),
                        numpy.array(delays)[k],
                    ]
                )
                cli.append(a)
            # Add the output to the two queues to communicate it with the main process
            queue_cl.put(cli)
            queue_v.put(vi)

        # Generate a list of subprocesses and of queues
        processes = []
        list_queue_cl = []
        list_queue_v = []
        for i in range(len(splitted_cell_indices)):
            # Each queue will be used by the subprocesses to communicate with the main process
            list_queue_cl.append(multiprocessing.Queue())
            list_queue_v.append(multiprocessing.Queue())
            proc = multiprocessing.Process(
                target=build_connector,
                args=(
                    splitted_cell_indices[i],
                    splitted_seeds[i],
                    list_queue_cl[i],
                    list_queue_v[i],
                ),
            )
            proc.start()
            processes.append(proc)

        # Gather the output of each subprocesses
        cl = []
        v = 0
        for i, p in enumerate(processes):
            cli = list_queue_cl[i].get()
            for clii in cli:
                cl.append(clii)
            v += list_queue_v[i].get()
            p.join()

        cl = numpy.hstack(cl).T
        method = self.sim.FromListConnector(cl)
        
        logger.warning("%s(%s): %g connections were created, %g per target neuron [%g]" % (self.name,self.__class__.__name__,len(cl),len(cl)/len(numpy.nonzero(self.target.pop._mask_local)[0]),v/len(numpy.nonzero(self.target.pop._mask_local)[0])))
	
        if len(cl) > 0:
            self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                method,
                                synapse_type=self.init_synaptic_mechanisms(),
                                label=self.name,
                                receptor_type=self.parameters.target_synapses)
        else:
            logger.warning("%s(%s): empty projection - pyNN projection not created." % (self.name,self.__class__.__name__))


class ModularSingleWeightProbabilisticConnector(ModularConnector):
    """
    ModularConnector that interprets the weights as proportional probabilities of connectivity.
    The parameter connection_probability is interepreted as the average probability that two neurons will be connected in this 
    projection. For each pair this connecter will make one random choice of connecting them (where the probability of this choice
    is determined as the proportional probability of the corresponding weight normalized by the connection_probability parameter).
    It will set each connections to the weight base_weight.
    """

    required_parameters = ParameterSet({
        'connection_probability': float,
        'base_weight' : PyNNDistribution
    })

    def _connect(self):
        cl = []
        for i in numpy.nonzero(self.target.pop._mask_local)[0]:
            weights = self._obtain_weights(i)
            delays = self._obtain_delays(i)
            conections_probabilities = weights/numpy.sum(weights)*self.parameters.connection_probability*len(weights)
            connection_indices = numpy.flatnonzero(conections_probabilities > numpy.random.rand(len(conections_probabilities)))
            cl.extend([(k,i,self.weight_scaler*self.parameters.base_weight.next(),delays[k]) for k in connection_indices])

        method = self.sim.FromListConnector(cl)
        logger.warning("%s: %g %g",self.name,min(conections_probabilities),max(conections_probabilities))
        logger.warning("%s: %d connections  [,%g,%g,%g]",self.name,len(cl),self.parameters.connection_probability,numpy.sum(weights),len(weights))
        
        if len(cl) > 0:
            self.proj = self.sim.Projection(
                                    self.source.pop,
                                    self.target.pop,
                                    method,
                                    synapse_type=self.init_synaptic_mechanisms(),
                                    label=self.name,
                                    receptor_type=self.parameters.target_synapses)
        else:
            logger.warning("%s(%s): empty projection - pyNN projection not created." % (self.name,self.__class__.__name__))
        




class ModularSamplingProbabilisticConnectorAnnotationSamplesCount(VariableNumSamplesConnector):
    """
    ModularConnector that interprets the weights as proportional probabilities of connectivity
    and for each neuron in connections it samples num_samples of
    connections that actually get realized according to these weights.
    Each such sample connections will have weight equal to
    base_weight but note that there can be multiple
    connections between a pair of neurons in this sample (in which case the
    weights are set to the multiple of the base weights times the number of
    occurrences in the sample).
    """

    required_parameters = ParameterSet({
        'num_samples': int,
        'base_weight' : PyNNDistribution,
        'annotation_reference_name': str,
    })

    def _connect(self):
        # Check if the delay function is incompatible with multiprocessing
        cl = []
        v = 0
        # Generates a splitted and of cells indices to be passed to each subprocesses
        seeds = mozaik.get_seeds(len(self.target.pop))[numpy.nonzero(self.target.pop._mask_local)[0]]

        splitted_seeds = numpy.array_split(seeds, int(self.model.num_threads))
        splitted_cell_indices = numpy.array_split(
            numpy.nonzero(self.target.pop._mask_local)[0], int(self.model.num_threads)
        )

        import multiprocessing

        # This code will be ran by each subprocess
        def build_connector(indices, seeds, queue_cl, queue_v):
            cli = []
            vi = 0

            for i in range(len(indices)):
                samples = self.target.get_neuron_annotation(
                    indices[i], self.parameters.annotation_reference_name
                )
                weights = self._obtain_weights(indices[i],seeds[i])
                # If a local module is defined, update the weights accordingly
                if self.parameters.local_module:
                    weights = self.local_module_weight_updates(indices[i], weights)

                delays = self._obtain_delays(indices[i],seeds[i])


                if self.parameters.num_samples == 0:
                    samples = self._obtain_num_samples(indices[i], samples)
                    co = Counter(
                        sample_from_bin_distribution(weights, int(samples), seeds[i])
                    )

                else:
                    assert self.parameters.num_samples > 2 * int(
                        samples
                    ), "%s: %d %d" % (
                        self.name,
                        self.parameters.num_samples,
                        2 * int(samples),
                    )
                    num_samples = self._obtain_num_samples(indices[i], self.parameters.num_samples - 2 * int(samples))
                    co = Counter(
                        sample_from_bin_distribution(
                            weights,
                            int(num_samples),
                            seeds[i],
                        )
                    )

                vi = vi + numpy.sum(list(co.values()))
                k = list(co.keys())
                a = numpy.array(
                    [
                        k,
                        numpy.zeros(len(k)) + indices[i],
                        self.weight_scaler
                        * numpy.multiply(
                            self.parameters.base_weight.copy(seeds[i]).next(len(k)),
                            list(co.values())
                        ),
                        numpy.array(delays)[k],
                    ]
                )
                cli.append(a)
            # Add the output to the two queues to communicate it with the main process
            queue_cl.put(cli)
            queue_v.put(vi)
        
        # Generate a list of subprocesses and of queues
        processes = []
        list_queue_cl = []
        list_queue_v = []
        for i in range(len(splitted_cell_indices)):
            # Each queue will be used by the subprocesses to communicate with the main process
            list_queue_cl.append(multiprocessing.Queue())
            list_queue_v.append(multiprocessing.Queue())
            proc = multiprocessing.Process(
                target=build_connector,
                args=(
                    splitted_cell_indices[i],
                    splitted_seeds[i],
                    list_queue_cl[i],
                    list_queue_v[i],
                ),
            )
            proc.start()
            processes.append(proc)

        # Gather the output of each subprocesses
        cl = []
        v = 0
        for i, p in enumerate(processes):
            cli = list_queue_cl[i].get()
            for clii in cli:
                cl.append(clii)
            v += list_queue_v[i].get()
            p.join()

        cl = numpy.hstack(cl).T
        method = self.sim.FromListConnector(cl)
        
        logger.warning("%s(%s): %g connections were created, %g per target neuron [%g]" % (self.name,self.__class__.__name__,len(cl),len(cl)/len(numpy.nonzero(self.target.pop._mask_local)[0]),v/len(numpy.nonzero(self.target.pop._mask_local)[0])))
        
        if len(cl) > 0:
            self.proj = self.sim.Projection(
                                self.source.pop,
                                self.target.pop,
                                method,
                                synapse_type=self.init_synaptic_mechanisms(),
                                label=self.name,
                                receptor_type=self.parameters.target_synapses)
        else:
            logger.warning("%s(%s): empty projection - pyNN projection not created." % (self.name,self.__class__.__name__))
