================
Making a release
================

To make a release of Mozaik requires you to have permissions to upload Mozaik
packages to the `Python Package Index`_ and the INCF Software Center, and to
upload documentation to the neuralensemble.org server.

When you think a release is ready, run through the following checklist one
last time:

    * do all the tests pass? You should do this on at least two Linux systems -- one a very
      recent version and one at least a year old. You should also do this with Python 3.7.
    * do all the example scripts generate the correct output? Run the
    * does the documentation build without errors? You should then at least skim
      the generated HTML pages to check for obvious problems.
    * have you updated the version numbers in :file:`setup.py`, :file:`src/__init__.py`,
      :file:`doc/conf.py` and :file:`doc/index.txt`?
    * have you updated the changelog?

Once you've confirmed all the above, create a source package using::

    $ python setup.py sdist

and check that it installs properly.

Now you should commit any changes, then tag with the release number as follows::

    $ git tag x.y.z

where ``x.y.z`` is the release number. You should now upload the documentation
to http://neuralensemble.org/docs/mozaik/.

.. todo:: more details on this

If this is a development release (i.e. an *alpha* or *beta*), the final step is
to upload the source package to the INCF Software Center [more instructions needed here].
Do **not** upload development releases to PyPI.

.. todo:: more details on this

If this is a final release, there are a few more steps:

    * if it is a major release (i.e. an ``x.y.0`` release), create a new bug-fix
      branch::

        $ git branch x.y

    * upload the source package to PyPI::

        $ python setup.py sdist upload

    * make an announcement on the `mailing list`_

    * if it is a major release, write a blog post about it with a focus on the
      new features and major changes

    * go home, take a headache pill and lie down for a while in a darkened room (-;

.. _`Python Package Index`: http://pypi.python.org/
.. _`mailing list`: http://groups.google.com/group/neuralensemble
