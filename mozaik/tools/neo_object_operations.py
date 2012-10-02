"""
This file contains various operations over Neo objects. Such as sum over lists
of Neo objects etc.
"""


def neo_sum(l):
    """
    This function gets a list of Neo objects and it
    adds them up. Importantly unlike Python sum function
    it starts adding to the first element of the list no to 0.
    """
    a = l[0]
    for z in l[1:]:
        a = a + z
    return a


def neo_mean(l):
    """
    Calculates the mean over list of Neo objects.
    See neo_sum for more details.
    """
    return neo_sum(l) / len(l)
