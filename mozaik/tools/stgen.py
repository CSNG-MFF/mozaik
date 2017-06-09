"""
This file contains code extracted from the NeuroTools package (http://neuralensemble.org/NeuroTools).
"""
from numpy import array, log
import numpy


class SpikeTrain(object):
    """
    SpikeTrain(spikes_times, t_start=None, t_stop=None)
    This class defines a spike train as a list of times events.

    Event times are given in a list (sparse representation) in milliseconds.

    Inputs:
        spike_times - a list/numpy array of spike times (in milliseconds)
        t_start     - beginning of the SpikeTrain (if not, this is infered)
        t_stop      - end of the SpikeTrain (if not, this is infered)

    Examples:
        >> s1 = SpikeTrain([0.0, 0.1, 0.2, 0.5])
        >> s1.isi()
            array([ 0.1,  0.1,  0.3])
        >> s1.mean_rate()
            8.0
        >> s1.cv_isi()
            0.565685424949
    """

    #######################################################################
    ## Constructor and key methods to manipulate the SpikeTrain objects  ##
    #######################################################################
    def __init__(self, spike_times, t_start=None, t_stop=None):
        #TODO: add information about sampling rate at time of creation

        """
        Constructor of the SpikeTrain object

        See also
            SpikeTrain
        """

        self.t_start     = t_start
        self.t_stop      = t_stop
        self.spike_times = numpy.array(spike_times, numpy.float32)

        # If t_start is not None, we resize the spike_train keeping only
        # the spikes with t >= t_start
        if self.t_start is not None:
            self.spike_times = numpy.extract((self.spike_times >= self.t_start), self.spike_times)

        # If t_stop is not None, we resize the spike_train keeping only
        # the spikes with t <= t_stop
        if self.t_stop is not None:
            self.spike_times = numpy.extract((self.spike_times <= self.t_stop), self.spike_times)

        # We sort the spike_times. May be slower, but is necessary by the way for quite a
        # lot of methods...
        self.spike_times = numpy.sort(self.spike_times, kind="quicksort")
        # Here we deal with the t_start and t_stop values if the SpikeTrain
        # is empty, with only one element or several elements, if we
        # need to guess t_start and t_stop
        # no element : t_start = 0, t_stop = 0.1
        # 1 element  : t_start = time, t_stop = time + 0.1
        # several    : t_start = min(time), t_stop = max(time)

        size = len(self.spike_times)
        if size == 0:
            if self.t_start is None:
                self.t_start = 0
            if self.t_stop is None:
                self.t_stop  = 0.1
        elif size == 1: # spike list may be empty
            if self.t_start is None:
                self.t_start = self.spike_times[0]
            if self.t_stop is None:
                self.t_stop = self.spike_times[0] + 0.1
        elif size > 1:
            if self.t_start is None:
                self.t_start = numpy.min(self.spike_times)
            if numpy.any(self.spike_times < self.t_start):
                raise ValueError("Spike times must not be less than t_start")
            if self.t_stop is None:
                self.t_stop = numpy.max(self.spike_times)
            if numpy.any(self.spike_times > self.t_stop):
                raise ValueError("Spike times must not be greater than t_stop")

        if self.t_start >= self.t_stop :
            raise Exception("Incompatible time interval : t_start = %s, t_stop = %s" % (self.t_start, self.t_stop))
        if self.t_start < 0:
            raise ValueError("t_start must not be negative")
        if numpy.any(self.spike_times < 0):
            raise ValueError("Spike times must not be negative")

    def __str__(self):
        return str(self.spike_times)

    def __del__(self):
        del self.spike_times

    def __len__(self):
        return len(self.spike_times)

    def __getslice__(self, i, j):
        """
        Return a sublist of the spike_times vector of the SpikeTrain
        """
        return self.spike_times[i:j]

    def is_equal(self, spktrain):
        """
        Return True if the SpikeTrain object is equal to one other SpikeTrain, i.e
        if they have same time parameters and same spikes_times

        Inputs:
            spktrain - A SpikeTrain object

        See also:
            time_parameters()
        """
        test = (self.time_parameters() == spktrain.time_parameters())
        return numpy.all(self.spike_times == spktrain.spike_times) and test

    def copy(self):
        """
        Return a copy of the SpikeTrain object
        """
        return SpikeTrain(self.spike_times, self.t_start, self.t_stop)


    def duration(self):
        """
        Return the duration of the SpikeTrain
        """
        return self.t_stop - self.t_start




class StGen:

    def __init__(self, rng=None, seed=None):
        """ 
        Stochastic Process Generator
        ============================

        Object to generate stochastic processes of various kinds
        and return them as SpikeTrain object.
      

        Inputs:
        -------
            rng - The random number generator state object (optional). Can be None, or 
                  a numpy.random.RandomState object, or an object with the same 
                  interface.

            seed - A seed for the rng (optional).

        If rng is not None, the provided rng will be used to generate random numbers, 
        otherwise StGen will create its own random number generator.
        If a seed is provided, it is passed to rng.seed(seed)

        Examples
        --------
            >> x = StGen()



        StGen Methods:
        ==============

        Spiking point processes:
        ------------------------
 
        poisson_generator - homogeneous Poisson process
        inh_poisson_generator - inhomogeneous Poisson process (time varying rate)
        """

        if rng==None:
            self.rng = numpy.random.RandomState()
        else:
            self.rng = rng

        if seed != None:
            self.rng.seed(seed)
        self.dep_checked = False

    def seed(self,seed):
        """ seed the gsl rng with a given seed """
        self.rng.seed(seed)


    def poisson_generator(self, rate, t_start=0.0, t_stop=1000.0, array=False,debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a Poisson process
        with the given rate (Hz) and stopping time t_stop (milliseconds).

        Note: t_start is always 0.0, thus all realizations are as if 
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        Inputs:
        -------
            rate    - the rate of the discharge (in Hz)
            t_start - the beginning of the SpikeTrain (in ms)
            t_stop  - the end of the SpikeTrain (in ms)
            array   - if True, a numpy array of sorted spikes is returned,
                      rather than a SpikeTrain object.

        Examples:
        --------
            >> gen.poisson_generator(50, 0, 1000)
            >> gen.poisson_generator(20, 5000, 10000, array=True)

        See also:
        --------
            inh_poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
        """

        #number = int((t_stop-t_start)/1000.0*2.0*rate)

        # less wasteful than double length method above
        n = (t_stop-t_start)/1000.0*rate
        number = numpy.ceil(n+3*numpy.sqrt(n))
        if number<100:
            number = min(5+numpy.ceil(2*n),100)

        if number > 0:
            isi = self.rng.exponential(int(1.0/rate), int(number))*1000.0
            if number > 1:
                spikes = numpy.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = numpy.array([])

        spikes+=t_start
        i = numpy.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i==len(spikes):
            # ISI buf overrun
            
            t_last = spikes[-1] + self.rng.exponential(1.0/rate, 1)[0]*1000.0

            while (t_last<t_stop):
                extra_spikes.append(t_last)
                t_last += self.rng.exponential(1.0/rate, 1)[0]*1000.0
            
            spikes = numpy.concatenate((spikes,extra_spikes))

            if debug:
                print "ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),len(extra_spikes))


        else:
            spikes = numpy.resize(spikes,(i,))

        if not array:
            spikes = SpikeTrain(spikes, t_start=t_start,t_stop=t_stop)


        if debug:
            return spikes, extra_spikes
        else:
            return spikes

            
    def inh_poisson_generator(self, rate, t, t_stop, array=False):
        """
        Returns a SpikeTrain whose spikes are a realization of an inhomogeneous 
        poisson process (dynamic rate). The implementation uses the thinning 
        method, as presented in the references.

        Inputs:
        -------
            rate   - an array of the rates (Hz) where rate[i] is active on interval 
                     [t[i],t[i+1]]
            t      - an array specifying the time bins (in milliseconds) at which to 
                     specify the rate
            t_stop - length of time to simulate process (in ms)
            array  - if True, a numpy array of sorted spikes is returned,
                     rather than a SpikeList object.

        Note:
        -----
            t_start=t[0]

        References:
        -----------

        Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
        Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
        Neural Comput. 2007 19: 2958-3010.

        Devroye, L. (1986). Non-uniform random variate generation. New York: Springer-Verlag.

        Examples:
        --------
            >> time = arange(0,1000)
            >> stgen.inh_poisson_generator(time,sin(time), 1000)

        See also:
        --------
            poisson_generator
        """

        if numpy.shape(t)!=numpy.shape(rate):
            raise ValueError('shape mismatch: t,rate must be of the same shape')

        # get max rate and generate poisson process to be thinned
        rmax = numpy.max(rate)
        ps = self.poisson_generator(rmax, t_start=t[0], t_stop=t_stop, array=True)

        # return empty if no spikes
        if len(ps) == 0:
            if array:
                return numpy.array([])
            else:
                return SpikeTrain(numpy.array([]), t_start=t[0],t_stop=t_stop)
        
        # gen uniform rand on 0,1 for each spike
        rn = numpy.array(self.rng.uniform(0, 1, len(ps)))

        # instantaneous rate for each spike
        
        idx=numpy.searchsorted(t,ps)-1
        spike_rate = rate[idx]

        # thin and return spikes
        spike_train = ps[rn<spike_rate/rmax]

        if array:
            return spike_train

        return SpikeTrain(spike_train, t_start=t[0],t_stop=t_stop)



