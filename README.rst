========================
Installation intructions
========================

Dependencies
============
* scipy/numpy
* nest (latest release, compiled with mpi)
* mpi4py
* pyNN (neo_output branch)
* imagen
* parameters
* quantities 
* neo

Installation
============
git clone https://github.com/antolikjan/mozaik.git

cd mozaik

python setup.py install


Detailed instructions
=====================

Mozaik requires currently some "non-standard" branches of software like the
pyNN which will be resolved in the future. Therefore more detailed installation
instructions follow.

Virtual env
___________

We recommended to install mozaik using the virtualenv python environment manager (http://pypi.python.org/pypi/virtualenv/) , to prevent potential
conflicts with standard versions of required libraries. Users can follow for example http://simononsoftware.com/virtualenv-tutorial/short tutorial or just do the following steps:
 
 * Install virtualenv
 * Create (for example in your home directory) a directory where all virtual
   environments will be created home/virt_env
 * Create the virtual environment for mozaik: virtualenv virt_env/virt_env_mozaik/ --verbose --no-site-packages

Then, load the virtual environment for mozaik by source virt_env/virt_env_mozaik/bin/activate

Your shell should look now something like:
(virt_env_mozaik)Username@Machinename:~$

Dependencies 
____________

 * scipy
 
   * scipy requires the following packages if you install it 'by hand' in your virtual environment: liblapack-dev, libblas-dev, gfortran
 
 * numpy
 * mpi4py
 * matplotlib (1.1 and higher)
 * quantities
 * PyNN:
     * git clone https://github.com/NeuralEnsemble/PyNN.git
     * Then, in your virtual environment: 
     * python setup.py install
 * Neo:
    * For Neo, you need to clone with the help of git:
    * git clone https://github.com/apdavison/python-neo python-neo
    * cd python-neo
    * python setup.py install
 * imagen:        
 
      * pip install --user imagen
 * parameters:
 
     * git clone https://github.com/apdavison/parameters.git parameters
     * cd parameters
     * python setup.py install
 * NeuroTools:
 
   * svn co https://neuralensemble.org/svn/NeuroTools/trunk NeuroTools
   * In virt_env_mozaik: python setup.py install
 
For mozaik itself, you need to clone with the help of git:
git clone https://github.com/antolikjan/mozaik.git

python setup.py install


VIRTUALENV NOTE: You might already have some of the above packages
if you've used the option --system-site-packages when creating the virtual environment for mozaik.
You can list the packages you have e.g. with the help of yolk):
If you've set up the virt_env with the option --system-site-packages and
you're using scipy, numpy, matplotlib anyway you don't have to install those in yout virt_env.


Ubuntu
======
Following these instruction should give you a working copy of mozaik on a 
fresh installation of Ubuntu (at the time of the writing the version was 12.04)

First the list of ubuntu package dependencies::

$ sudo apt-get install python2.7 python-dev python-pip python-nose subversion git libopenmpi-dev g++ libjpeg8 libjpeg8-dev libfreetype6 libfreetype6-dev zlib1g-dev libpng++-dev libncurses5 libncurses5-dev libreadline-dev liblapack-dev libblas-dev gfortran libgsl0-dev

Then python virtualenv and virtualenvwrapper (an handy way to manage python virtual environments)::

$ sudo pip install virtualenv
$ sudo pip install virtualenvwrapper

To setup `virtualenvwrapper <http://virtualenvwrapper.readthedocs.org/en/latest//>`_ add the following lines at the top of ~/.bash_profile (create it if you don't have one)::

    # virtualenvwrapper
    export WORKON_HOME=~/.virtualenvs
    source /usr/local/bin/virtualenvwrapper.sh
    export PIP_VIRTUALENV_BASE=$WORKON_HOME
    export PIP_RESPECT_VIRTUALENV=true

For the first time, run bash_profile (the next times it will be loaded by your terminal)::      

$ source .bash_profile

To create a new managed virtualenv you just need to::

    $ mkvirtualenv --no-site-packages mozaik
    $ workon mozaik
    (mozaik)$>
 
To produce a requirement file::

(mozaik)$> pip freeze > requirements.txt
 
Then you can use it to replicate installation::

(mozaik)$> pip install -r requirements.txt
 
To exit the environment::

(mozaik)$> deactivate
 
Now you can install in this protected environment all other dependencies::

(mozaik)$ pip install numpy mpi4py scipy matplotlib quantities lazyarray

Now we can install *Nest*:

    - download the latest version from their `website <http://www.nest-initiative.org/index.php/Software:Download>`_
    - untar and cd into it::

        (mozaik)$ tar xvfz nest-2.2.2.tar.gz
        (mozaik)$ cd nest-2.2.2
    - then configure, choose if you want mpi. And, if you decide to have nest installed somewhere else from normal places add it with a prefix, then you also need to specify the pynest prefix::
    
        (mozaik)$ ./configure (--with-mpi) (--prefix=$HOME/opt/nest --with-pynest-prefix=$WORKON_HOME/mozaik)
    - finally make and install (with sudo, since nest has also other places to install its own stuff)::

        (mozaik)$ make
        (mozaik)$ sudo make install
    - in the ./~nestrc uncomment the lines regarding mpirun, then::

        (mozaik)$ sudo make installcheck

Install PyNN::

    $ git clone https://github.com/NeuralEnsemble/PyNN.git
    (mozaik)$ cd PyNN/
    (mozaik)$ python setup.py install

that will reside into::
    
    /home/.../.virtualenvs/mozaik/lib/python2.7/site-packages/PyNN-0.8dev-py2.7.egg-info

Install NEO::

    $ git clone https://github.com/apdavison/python-neo python-neo
    (mozaik)$ cd python-neo/
    (mozaik)$ python setup.py install

Install Imagen::
    
    (mozaik)$ pip install imagen

Install Parameters package::

    $ git clone https://github.com/apdavison/parameters.git parameters
    (mozaik)$ cd parameters/
    (mozaik)$ python setup.py install

Install NeuroTools::

    $ svn co https://neuralensemble.org/svn/NeuroTools/trunk NeuroTools
    (mozaik)$ cd NeuroTools/
    (mozaik)$ python setup.py install

And, finally, Mozaik::
    
    $ git clone https://github.com/antolikjan/mozaik.git
    (mozaik)$ cd mozaik/
    (mozaik)$ python setup.py install

:copyright: Copyright 2011-2013 by the *mozaik* team, see AUTHORS.
:license: `CECILL <http://www.cecill.info/>`_, see LICENSE for details.
