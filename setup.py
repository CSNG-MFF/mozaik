#!/usr/bin/env python
from distutils.core import setup
      
setup(
    name = "mozaik",
    version = "0.1dev",
    package_dir = {'mozaik': 'mozaik'},
    packages = ['mozaik',
                'mozaik.analysis',
                'mozaik.framework',
                'mozaik.models',
                'mozaik.models.retinal',
                'mozaik.stimuli',
                'mozaik.storage',
                'mozaik.tools',
                'mozaik.visualization'
                ],
    author = "The Mozaik team",
    author_email = "antolikjan@gmail.com",
    description = "Python package mozaik is an integrated workflow framework for large scale neural simulations.",
    long_description = open('README.md').read(),
    license = "CeCILL http://www.cecill.info",
    keywords = "computational neuroscience simulation large-scale model spiking",
    url = "http://neuralensemble.org/mozaik",
    classifiers = ['Development Status :: 2 - Pre-Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: Other/Proprietary License',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering'],
)

