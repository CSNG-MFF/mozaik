"""
This file contains various operations over neo objects. Such as sum over lists of neo objects etc.
"""

def neo_sum(l):
    """
    This function gets a list of neo objects and it
    adds them up. Importantly unlike python sum function
    it starts adding to the first element of the list no to 0.
    """
    a = l[0]
    for z in l[1:]:
        a = a + z
    return a

def neo_mean(l):
    """
    Calculates the mean over list of neo objects. 
    See neo_sum for more details.
    """
    return neo_sum(l)/len(l)
