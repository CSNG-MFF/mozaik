# encoding: utf-8
from pyNN.common import Population
from pyNN.errors import NothingToWriteError
import pyNN.recording.files
from MozaikLite.framework.space import VisualRegion
import logging
from string import Template

logger = logging.getLogger("MozaikLite")


class Layer(object):        
    """
    A Layer object represents a neuronal structure having the form of a lamina,
    e.g. a cortical layer.
    It contains a number of Population objects, each representing one of the
    cell types found within that lamina.
    """

    def __init__(self, population_list, label):
        self.populations = {}
        self.label = label
        for p in population_list:
            self.add_population(p) 

    def add_population(self, population):
        logger.debug('Adding population "%s", consisting of %s %s neurons, to layer "%s"' % (population.label,
                                                                                             population.size,
                                                                                             population.celltype.__class__.__name__,
                                                                                             self.label))
        assert isinstance(population, Population), "%s is not a Population object" % p
        if population.label is None:
            raise Exception("Populations must have labels.")
        else:
            self.populations[population.label] = population
    
    def record(self, variable, cells='all'):
        func_mapping = {'spikes': 'record', 'v': 'record_v'} # need to add conductances
        record_method = func_mapping[variable]
        for name, p in self.populations.items():
            if cells == 'all':
                logger.debug('Recording %s from all cells in population "%s" in layer "%s"' % (variable, name, self.label))
                getattr(p, record_method)()
            elif isinstance(cells, dict):
                logger.debug('Recording %s from a subset of cells in population "%s" in layer "%s"' % (variable, name, self.label))
                getattr(p, record_method)(cells[name])
            elif isinstance(cells, int):
                n = cells
                logger.debug('Recording %s from a subset of %d cells in population "%s" in layer "%s"' % (variable, n, name, self.label))
                getattr(p, record_method)(n)
            else:
                raise Exception("cells must be 'all', a dict, or an int. Actual value of %s" % str(cells))
    
    def write(self, path, file_type=pyNN.recording.files.StandardTextFile):
        self.filenames = {}
        self.file_type = file_type
        for p in self.populations.values():
            spike_file = file_type("%s_%s_%s.spikes" % (path, self.label.replace(" ","-"), p.label), mode='w')
            vm_file = file_type("%s_%s_%s.v" % (path, self.label.replace(" ","-"), p.label), mode='w')
            self.filenames[p.label] = {}
            try:
                p.printSpikes(spike_file)
                self.filenames[p.label]['spikes'] = spike_file.name
                logging.debug("Writing spikes from population %s to file %s." % (p, spike_file))
            except NothingToWriteError, errmsg: 
                logger.debug(errmsg)
            try:
                p.print_v(vm_file)
                self.filenames[p.label]['v'] = vm_file.name
                logging.debug("Writing Vm from population %s to file %s." % (p, vm_file))
            except NothingToWriteError, errmsg: 
                logger.debug(errmsg)
        return self.filenames
            
    def describe(self, template='default',
                 render=lambda t,c: Template(t).safe_substitute(c)):
        if template == 'default':
            template = """Layer "$label" consisting of the following populations:\n$populations."""
        context = {
            'label': self.label,
            'populations': [p.describe(template=None) for p in self.populations.values()]
        }
        if template:
            return render(template, context)
        else:
            return context
    
            
