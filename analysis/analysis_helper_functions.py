from MozaikLite.stimuli.stimulus_generator import parse_stimuls_id

def _colapse(dd,axis):
    d = {}
    for s in dd:
        s1 = parse_stimuls_id(s)
        s1.parameters[axis]='*'
        s1 = str(s1)
        if d.has_key(s1):
           d[s1].append(dd[s])
        else:
           d[s1] = dd[s]
    return d
    
def colapse(value_list,stimuli_list,parameter_indexes=[]):
    ## it colapses the value_list acording to stimuli with the same value 
    ## of parameters whose indexes are listed in the <parameter_indexes> and 
    ## replaces the collapsed parameters in the 
    ## stimuli_list with *
    d = {}
    for v,s in zip(value_list,stimuli_list):
        d[str(s)]=[v]

    for ax in parameter_indexes:
        d = _colapse(d,ax)
    
    return (d.values(),d.keys())
