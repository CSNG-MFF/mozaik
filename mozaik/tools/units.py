"""
This module defines units used in mozaik project and not specified in the quantities package.
"""
import quantities as qt
from quantities.unitquantity import UnitQuantity, UnitInformation
import numpy

sp = spike = UnitInformation(
    'spike',
    symbol='sp',
    aliases=['spike', 'sp']
)

spike_per_sec = UnitQuantity(
    'spike_per_sec',
    spike/qt.s,
    symbol='sp/s',
)

#defines cycles per radian of visual field angle
cpr = UnitQuantity(
    'cycles_per_radian',
    qt.cycle/qt.rad
)

#defines cycles per degree of visual field
cpd = UnitQuantity(
    'cycles_per_degree',
    qt.cycle/qt.degree
)

lux = lx = UnitQuantity(
    'lux',  # I've seen both luxes and luces given as plurals, but I think in the SI unit sense, "lux" is both singular and plural - see Oxford English Dictionary
    symbol='lx',
    aliases=['lux', 'lx']
)

nS = nanosiemens = UnitQuantity(
    'nanosiemens',
    1e-9*qt.S,
    symbol='nS',
    aliases=['nanosiemens']
)


def periodic(unit):
    """
    Checks whether a units is periodic

    Returns
    -------
        (a,b): bool,double
             Where a is True if unit is periodic, and b corresponds to the period if a is True.
    """
    periodic = False
    period = None

    if unit == qt.rad:
        periodic = True
        period = 2*numpy.pi
    elif unit == qt.degrees:
        periodic = True
        period = 360

    return (periodic, period)
