"""
Units used in mozaik project and not specified in quantities.
"""
import quantities as qt
from quantities.unitquantity import UnitQuantity, UnitInformation, dimensionless
from quantities import s
import numpy

sp = spike = UnitInformation(
    'spike',
    symbol='sp',
    aliases = ['spike', 'sp']
)

spike_per_sec = UnitQuantity(
    'spike_per_sec',
    spike/s,
    symbol='sp/s',
)



def periodic(unit):
    """
    Checks whether a units is periodic
    
    return (a,b) where a is True if unit is periodic, and b corresponds to the period if a is True
    """
    periodic = False
    period = None
    
    if unit == qt.rad:
       periodic = True
       period = 2*numpy.pi
    elif unit == qt.degrees:
       periodic = True
       period = 360
    
    return (periodic,period)