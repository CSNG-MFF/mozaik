analysis Package
================

This sub-package contains modules that are related to analysis.

Modules
^^^^^^^    
    * :py:mod:`MozaikLite.analysis.analysis` - contains definition of the analysis interface and implementations of several analysis codes
    * :py:mod:`MozaikLite.analysis.analysis_data_structures` - contains definitions of data stractures that the analysis algorithms export
    * :py:mod:`MozaikLite.analysis.analysis_helper_functions` - some additional functions that are often used in the analysis package 
    
Architecture
^^^^^^^^^^^^

The :py:class:`MozaikLite.analysis.analysis.Analysis` interface assumes that the input to the analysis is retrieved from 
:py:class:`MozaikLite.storage.datastore`, in the form of recorded data or analysis data structures (see :py:mod:`MozaikLite.analysis.analysis_data_structures`)
produced by other analysis algorithms that were run previously. The analysis itself should be able to filter 
out from the datastore the recordings/analysis_data_structures that it can handle and should be as encompassing and general as possible.
This way it will be possible to significantly 'configure' the analysis process by merely filtering the data given 
to the Analysis classes via the :py:mod:`MozaikLite.storage.queries` mechanisms. After the analysis is done the results are 
packaged in one of the data structures defined in :py:mod:`MozaikLite.analysis.analysis_data_structures` and sent to the 
datastorage. Thus when creating new analysis one should always check which data structures are already defined, and define
a new one only in case it is not possible to map the new results data structure on any of the existing ones.




:mod:`analysis` Module
----------------------

.. automodule:: MozaikLite.analysis.analysis
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`analysis_data_structures` Module
--------------------------------------

.. automodule:: MozaikLite.analysis.analysis_data_structures
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`analysis_helper_functions` Module
---------------------------------------

.. automodule:: MozaikLite.analysis.analysis_helper_functions
    :members:
    :undoc-members:
    :show-inheritance:

