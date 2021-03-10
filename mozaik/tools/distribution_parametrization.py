"""
This module contains code interfacing parameters package, and pyNN distribution interface. 

In future pyNN plans to make an comprehensive merge between the parameters parametrization system and pyNN,
in which case this code should become obsolete and mozaik should fully switch to such new system.
"""

from parameters import ParameterSet, ParameterRange, ParameterTable, ParameterReference
from pyNN.random import RandomDistribution, NumpyRNG
import urllib, copy, warnings, numpy, numpy.random  # to be replaced with srblib
from urlparse import urlparse

import mozaik

def load_parameters(parameter_url,modified_parameters):
    """
    A simple function for loading parameters that replaces the values in *modified_parameters* in the loaded parameters
    and subsequently expands references.
    """
    parameters = MozaikExtendedParameterSet(parameter_url)
    parameters.replace_values(**modified_parameters)
    parameters.replace_references()
    return parameters

class PyNNDistribution(RandomDistribution):
      """
      This will be the wraper for the PyNN RandomDistribution
      
      The first parameter is the name of the distribution (see pyNN.random.RandomDistribution)
      The params is a tuple of parameters of the corresponding numpy distribution (see pyNN.random.RandomDistribution)
      For the rest of the parameters see pyNN.random.RandomDistribution
      """
      def __init__(self,name,**params):
          self._first = True
          RandomDistribution.__init__(self,name,rng=mozaik.pynn_rng,**params)

      def __str__(self):
          ps = ','.join([str(k) + '=' + str(self.parameters[k]) for k in self.parameters.keys()])
          return "PyNNDistribution(name=" + '\'' + self.name + '\',' +  ps + ')'


class ParameterWithUnitsAndPeriod():
    """
    This is a parameter that allows us add Units and Period to a given parameter.
    """
    def __init__(self, value,units=None,period=None):
        self.value = value
        self.units = units
        self.period = period

    def __repr__(self):
        return "ParameterWithUnitsAndPeriod("+str(self.value)+",units=" + str(self.units) + ",period=" + str(self.period) + ")"
          
class MozaikExtendedParameterSet(ParameterSet):
    """
    This is an extension to `ParameterSet` class which adds the PyNNDistribution as a possible type of a parameter.
    """

    @staticmethod
    def read_from_str(s,update_namespace=None):
        global_dict = dict(ref=ParameterReference,url=MozaikExtendedParameterSet,ParameterSet=ParameterSet)
        global_dict.update(dict(ParameterRange=ParameterRange,
                                ParameterTable=ParameterTable,
                                PyNNDistribution = PyNNDistribution,
                                RandomDistribution = RandomDistribution,
                                NumpyRNG=NumpyRNG,
                                ParameterWithUnitsAndPeriod=ParameterWithUnitsAndPeriod,
                                pi=numpy.pi))
        if update_namespace:
            global_dict.update(update_namespace)
        
        D=None
        try:
            D = eval(s, global_dict)
        except SyntaxError as e:
            raise SyntaxError("Invalid string for ParameterSet definition: %s\n%s" % (s,e))
        except NameError as e:
            raise NameError("%s\n%s" % (s,e))
            
        return D or {}
    
    def __init__(self, initialiser, label=None, update_namespace=None):
        if update_namespace == None:
           update_namespace = {}
        update_namespace['PyNNDistribution'] = PyNNDistribution

        def walk(d, label):
            # Iterate through the dictionary `d`, replacing `dict`s by
            # `ParameterSet` objects.
            for k,v in d.items():
                ParameterSet.check_validity(k)
                if isinstance(v, ParameterSet):
                    d[k] = v
                elif isinstance(v, dict):
                    d[k] = walk(v, k)
                else:
                    d[k] = v
            return MozaikExtendedParameterSet(d, label)
        
        self._url = None
        if isinstance(initialiser, basestring): # url or str
            try:
                # can't handle cases where authentication is required
                # should be rewritten using urllib2 
                #scheme, netloc, path, \
                #        parameters, query, fragment = urlparse(initialiser)
                f = urllib.urlopen(initialiser)
                pstr = f.read()
                self._url = initialiser

                
            except IOError:
                pstr = initialiser
                self._url = None
            else:
                f.close()


            # is it a yaml url?
            if self._url:
                import urlparse, os.path
                o = urlparse.urlparse(self._url)
                base,ext = os.path.splitext(o.path)
                if ext in ['.yaml','.yml']:
                    import yaml
                    initialiser = yaml.load(pstr)
                else:
                    initialiser = MozaikExtendedParameterSet.read_from_str(pstr,update_namespace)
            else:
                initialiser = MozaikExtendedParameterSet.read_from_str(pstr,update_namespace)

        
        # By this stage, `initialiser` should be a dict. Iterate through it,
        # copying its contents into the current instance, and replacing dicts by
        # ParameterSet objects.
        if isinstance(initialiser, dict):
            for k,v in initialiser.items():
                ParameterSet.check_validity(k)
                if isinstance(v, ParameterSet):
                    self[k] = v
                elif isinstance(v, dict):
                    self[k] = walk(v, k)
                else:
                    self[k] = v
        else:
            raise TypeError("`initialiser` must be a `dict`, a `ParameterSet` object, a string, or a valid URL")

        # Set the label
        if hasattr(initialiser, 'label'):
            self.label = label or initialiser.label # if initialiser was a ParameterSet, keep the existing label if the label arg is None
        else:
            self.label = label
        
        # Define some aliases, allowing, e.g.:
        # for name, value in P.parameters():
        # for name in P.names():
        self.names = self.keys
        self.parameters = self.items
