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
     
     * PyNN requires currently the neo-output branch, NOT the standard one. So, you need to do the following: 
     * svn co https://neuralensemble.org/svn/PyNN/branches/neo_output/
     * Then, in your virtual environment: 
     * python setup.py install
 * Neo:
 
    * For Neo, you need to clone with the help of git:
    *  git clone https://github.com/apdavison/python-neo python-neo
    *  cd python-neo
    *  python setup.py install
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

Running tests
_____________

To run tests and measure code coverage, run

$ nosetests --with-coverage --cover-erase --cover-package=mozaik --cover-html --cover-inclusive

in the root directory of the Mozaik package


:copyright: Copyright 2011-2013 by the *mozaik* team, see AUTHORS.
:license: `CECILL <http://www.cecill.info/>`_, see LICENSE for details.
