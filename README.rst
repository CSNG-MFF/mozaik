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

  pip3 install numpy scipy mpi4py matplotlib quantities lazyarray interval Pillow param==1.5.1 parameters neo==0.12.0 cython psutil future requests elephant pytest-xdist pytest-timeout junitparser numba numpyencoder sphinx imageio scikit-image

Next we will manually install several packages. It is probably the best if you create a separate directory in an appropriate
place, where you will download and install the packages from.

First install the *imagen* package::

  git clone https://github.com/CSNG-MFF/imagen.git
  cd imagen
  pip install .

Then install the *PyNN* package from the PyNNStepCurrentModule branch::

  git clone https://github.com/CSNG-MFF/PyNN.git
  cd PyNN
  git checkout PyNNStepCurrentModule
  pip install .

Next install the *Nest* simulator (always in the virtual environment):

    - download the latest version from their `website <http://www.nest-initiative.org/index.php/Software:Download>`_::
        
        wget https://github.com/nest/nest-simulator/archive/refs/tags/v3.4.tar.gz
        
    - untar and cd into it::

        tar xvfz v3.4.tar.gz
        cd nest-simulator-3.4
    
    - then configure (change path to wherever you installed your virtual environemnt)::
    
        (mozaik)$ cmake -Dwith-mpi=ON -Dwith-boost=ON -DCMAKE_INSTALL_PREFIX:PATH=$HOME/virt_env/mozaik -Dwith-optimize='-O3' ./
       
    - finally, by launching make and install, it installs PyNest in the activated virtual environment mozaik. If you're using Slurm, run these commands through :code:`srun` ::

        (mozaik)$ make
        (mozaik)$ make install
        
    - Then::
        
        (mozaik)$ make installcheck

      or if you are using Slurm::

        (mozaik)$ salloc -n8 make installcheck
    
    - nest will reside in $HOME/virt_env/mozaik/lib/python3.*/site-packages. Check that the package is seen by python using::

        (mozaik)$ python -c 'import nest'

Then install the *stepcurrentmodule* Nest module:

    - get the module from github and cd into it::
        
        git clone https://github.com/CSNG-MFF/nest-step-current-module.git
        cd nest-step-current-module

    - then, in the following command, replace NEST_CONFIG_PATH by your nest-config installation path (should reside in $HOME/virt_env/mozaik/bin/nest-config) and run it::
        
        (mozaik)$ cmake -Dwith-mpi=ON -Dwith-boost=ON -Dwith-optimize='-O3' -Dwith-nest=NEST_CONFIG_PATH ./

    - finally, by launching make and install, it installs the nest module in the activated virtual environment mozaik. If you're using Slurm, run these commands through :code:`srun` ::

        (mozaik)$ make
        (mozaik)$ make install

    - Check that the package is seen by python using::

        (mozaik)$ python -c 'import nest; nest.Install("stepcurrentmodule")'

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

  pytest -m 'not mpi' && black --check .

This command will run all tests that it can find recursively under the current directory, as well as check all non-blacklisted files for formatting. Travis-CI will run the same steps for your pull request once you submit it to the project. To install pytest and black::

  pip3 install pytest pytest-cov pytest-randomly coverage black

Note that the mpi tests are currently not working when invoking pytest in this manner. You can run these specific tests the following way::

  pytest -m 'not not_github' tests/full_model/test_models_mpi.py

Due to the impossibility of using more than 2 cores in Github actions, the test :code:`test_mozaik_rng_mpi7` invoking 7 MPI processes cannot be ran there. Also, as it requires the allocation of 7 MPI slots it might not be possible to run in without slurm as the other MPI tests. It is therefore the responsibility of the contributor to run it locally through slurm before pushing changes::

  salloc -n7 pytest -m 'not_github' tests/full_model/test_models_mpi.py

There are additional useful options for pytests that you can use during development:

    - You may exclude tests running the model by adding the option::

        pytest -m "not model"

    - To avoid running the full size model, you can run a smaller version of it::

        pytest -m 'LSV1M_tiny'

    - You can run the tests in a single file by::

        pytest path/to/file

    - Pytest doesn't, print to :code:`stdout` by default, you can enable this by::

        pytest -s

:copyright: Copyright 2011-2013 by the *mozaik* team, see AUTHORS.
:license: `CECILL <http://www.cecill.info/>`_, see LICENSE for details.

