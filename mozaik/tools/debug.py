import pylab
import pickle

def visualize_retinal_cache(data_store,which):
    f = open('./retinal_cache/' + str(which) + '.st','r')
    cached_stimulus = pickle.load(f1)
    z = pickle.load(f)   
    
    pos = data_store.get_neuron_positions()['X_ON']
    
    pylab.figure()
    pylab.scatter(pos[0],pos[1],color=cached_stimulus[:,0])
