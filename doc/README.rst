Mozaik Documentation Setup
==========================

To build the documentation locally, follow these steps:

1. Install Mozaik and its dependencies according to the main instructions.

2. Install additional documentation dependencies:

   .. code-block:: bash

      pip install numpydoc
      sudo apt install octave
      pip install oct2py

   If you don’t have sudo privileges, install Octave via a container, module system, or local build.

3. Generate the documentation using Sphinx:

   .. code-block:: bash

      cd doc
      make html