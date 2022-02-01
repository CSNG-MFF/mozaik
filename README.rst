Mozaik
------

Computational neurosceince is shifting towards more heterogeneous models of neuronal circuits, and application of complex experimental protocols. This escalation of complexity is not sufficiently met by existing tool chains. **Mozaik** is a workflow system for spiking neuronal network simulations written in Python that integrates model, experiment and stimulation specification, simulation execution, data storage, data analysis and visualization into a single automated workflow. This way, **Mozaik** increases the productivity of running virtual experiments on complex heterogenous spiking neuronal networks. 

You can read more about **Mozaik** `here <https://www.frontiersin.org/articles/10.3389/fninf.2013.00034/full>`_.

Currently, **Mozaik** is being fully tested only on Ubuntu Linux distribution.

Ubuntu installation instructions
--------------------------------

Following these instruction should give you a working copy of mozaik on a 
fresh installation of current Ubuntu system.

First the list of ubuntu package dependencies::

  sudo apt-get install python3 python3-dev python3-pip python3-setuptools python3-tk python-nose subversion git libopenmpi-dev g++ libjpeg8 libjpeg8-dev libfreetype6 libfreetype6-dev zlib1g-dev libpng++-dev libncurses5 libncurses5-dev libreadline-dev liblapack-dev libblas-dev gfortran libgsl0-dev openmpi-bin python-tk cmake libboost-all-dev


Virtual env
____________

Then python virtualenv and virtualenvwrapper (a handy way to manage python virtual environments)::

$ sudo pip3 install virtualenv
$ sudo pip3 install virtualenvwrapper

To setup `virtualenvwrapper <http://virtualenvwrapper.readthedocs.org/en/latest//>`_ add the following lines at the top of ~/.bashrc ::

    # virtualenvwrapper
    export WORKON_HOME=~/virt_env
    source /usr/local/bin/virtualenvwrapper.sh
    export PIP_VIRTUALENV_BASE=$WORKON_HOME
    export PIP_RESPECT_VIRTUALENV=true

For the first time, run .bashrc (in the future it will be loaded by your terminal)::      

$ source .bashrc

To create a new managed virtualenv you just need to::

    $ mkvirtualenv --python=/usr/bin/python3 mozaik
    $ workon mozaik
    (mozaik)$>
 

Dependencies 
____________

 
Now you can install all other dependencies in this protected environment::

  pip3 install numpy scipy mpi4py matplotlib quantities lazyarray interval Pillow param==1.5.1 parameters neo==0.9.0 cython pynn psutil future requests elephant

Next we will manually install several packages. It is probably the best if you create a separate directory in an appropriate
place, where you will download and install the packages from.

First install the *imagen* package::

  git clone https://github.com/CSNG-MFF/imagen.git
  cd imagen
  python setup.py install

Next install the *Nest* simulator (always in the virtual environment):

    - download the latest version from their `website <http://www.nest-initiative.org/index.php/Software:Download>`_
        
        wget https://github.com/nest/nest-simulator/archive/v2.20.1.tar.gz
        
    - untar and cd into it::

        tar xvfz v2.20.1.tar.gz
        cd nest-simulator-2.20.1
    
    - then configure (change path to wherever you installed your virtual environemnt)::
    
        (mozaik)$ cmake -Dwith-mpi=OFF -Dwith-boost=ON -DCMAKE_INSTALL_PREFIX:PATH=$HOME/virt_env/mozaik -Dwith-optimize='-O3' ./
       
    - finally, by launching make and install, it installs PyNest in the activated virtual environment mozaik::
    
        (mozaik)$ make
        (mozaik)$ make install
        
    - Then::
        
        make installcheck
    
    - nest will reside in $HOME/virt_env/mozaik/lib/python3.*/site-packages. Check that the package is seen by python using::
        python -c 'import nest'


And, finally, Mozaik::
    
    git clone https://github.com/CSNG-MFF/mozaik.git
    cd mozaik
    python setup.py install
    

.. _ref-run:


Running examples
----------------

Go to the examples directory in the mozaik cloned from github (see above) and launch the model VogelsAbbott2005::

  cd examples
  cd VogelsAbbott2005
  python run.py nest 2 param/defaults 'test'
  
This will launch the example with the nest simulator running 2 MPI processes, each process running 2 threads, using the parameterization of the model rotted in param/defaults. Finally, 'test' is the name of this run.


Testing, Autoformat, Continuous Integration
-------------------------------------------

In case you want to contribute to the project, you need to make sure your code passes all unit tests and is formatted with the Black autoformatter. You can make sure this is the case by running following from the project directory::

  pytest && black --check .

This command will run all tests that it can find recursively under the current directory, as well as check all non-blacklisted files for formatting. Travis-CI will run the same steps for your pull request once you submit it to the project. To install pytest and black::

  pip3 install pytest pytest-cov pytest-randomly coverage black

There are additional useful options for pytests that you can use during development:

    - You may exclude tests running the model by adding the option::

        pytest -m "not model"
    - You can run the tests in a single file by::

        pytest path/to/file
    - Pytest doesn't, print to :code:`stdout` by default, you can enable this by::

        pytest -s

:copyright: Copyright 2011-2013 by the *mozaik* team, see AUTHORS.
:license: `CECILL <http://www.cecill.info/>`_, see LICENSE for details.

