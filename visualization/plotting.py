import pylab
import numpy
from MozaikLite.stimuli.stimulus_generator import parse_stimuls_id,load_from_string

class TuningCurve(object):
    """
    values - contain a list of lists of values, the outer list corresponding
    to stimuli the inner to neurons.
    
    stimuli_ids - contain liest of stimuli ids corresponding to the values
    
    parameter_index - corresponds to the parameter that should be plotted as 
                    - a tuning curve
                    
    neurons - which        """
    
    def  __init__(self,values,stimuli_ids,parameter_index):
        self.d = {}
       
        # creat dictionary where stimulus_id indexes all the different values and corresponding data for a given 
        # throught the range of the selected parameter
        
        for (v,s) in zip(values,stimuli_ids):
            s = parse_stimuls_id(s)
            val = s.parameters[parameter_index]
            s.parameters[parameter_index]='x'
            
            if self.d.has_key(str(s)):
               (a,b) = self.d[str(s)] 
               a.append(v)
               b.append(val)
            else:
               self.d[str(s)]  = ([v],[val]) 
        
        for k in self.d:
            (a,b) = self.d[k]
            self.d[k] = (numpy.array(a),b)
    
    def plot(self,neurons=[],ylabel=''):
        if neurons == []:
           neurons = numpy.arange(0,len(self.d[self.d.keys()[0]][0][0]),1) 

        for n in neurons:
            pylab.figure()
            for k in  self.d:
                (a,b) = self.d[k]
                par,val = zip(*sorted(zip(b,a[:,n])))
                pylab.plot(par,val,fromat_stimulus_id(parse_stimuls_id(k)))
            pylab.title('Orientation tuning curve, Neuron: %d' % n)
            
class CyclicTuningCurve(TuningCurve):
    """
    Tuning curve over cyclic domain
    """
    def plot(self,neurons=[],ylabel=''):
        if neurons == []:
           neurons = numpy.arange(0,len(self.d[self.d.keys()[0]][0][0]),1) 

        for n in neurons:
            pylab.figure()
            for k in  self.d:
                print 'CCCC:' + str(k)
                print self.d[k]
                (a,b) = self.d[k]
                par,val = zip(*sorted(zip(b,a[:,n])))
                # make the tuning curve to wrap around  
                par = list(par)
                val = list(val)
                par.append(par[0])
                val.append(val[0])
                pylab.plot(numpy.arange(len(val)),val,fromat_stimulus_id(parse_stimuls_id(k)))
            pylab.ylabel(ylabel)
            pylab.xticks(numpy.arange(len(val)),["%.2f"% float(a) for a in par])
            pylab.title('Orientation tuning curve, Neuron: %d' % n)
      
        
def fromat_stimulus_id(stimulus_id):
    string = ''
    for p in stimulus_id.parameters:
        if p != '*' and p != 'x':
            string = string + ' ' + str(p)
    return string

