Installation
============

In this section, we discuss ways to install ``vampire-analysis`` and
best practices.

Beginner
--------

``vampire-analysis`` runs with Python. If we don’t have Python yet, we
recommend to install `Anaconda`_, which includes Python and many
other packages for data science and scientific computing. Detailed
installation instructions can be found in `Anaconda documentation`_.

After installation of Anaconda, open the Anaconda command prompt and
install ``vampire-analysis`` via `PyPI`_ by typing:

::

   pip install vampire-analysis

We can now use ``vampire-analysis`` in the command prompt python
interpreter or Jupyter Notebooks.

Advanced
--------

Installing the package directly may cause dependency conflicts with
other packaged. To avoid such problem, we can create a `virtual
environment`_ by ``venv``. The following is a quick tutorial on creating
a virtual environment. For a detailed walk through, see YouTube videos
for `Windows`_ and `MacOS`_.

Suppose we have a project named ``my_project`` in Windows, we can create
a virtual environment by typing in the command line:

::

   python -m venv my_project\venv

We can activate the virtual environment with

::

   my_project\venv\Scripts\activate.bat

To install ``vampire-analysis``, use the above command

::

   pip install vampire-analysis

or for fully reproducible dependencies, use

::

   pip install -r requirements.txt

where ``requirement.txt`` of this project is placed in the current
working directory.

We can verify the environment only has necessary packages installed with
the command

::

   pip list

To exit the environment, simply type

::

   deactivate

.. _Anaconda: https://www.anaconda.com/products/individual
.. _Anaconda documentation: https://docs.anaconda.com/anaconda/install/
.. _PyPI: https://pypi.org/project/vampire-analysis/
.. _virtual environment: https://docs.python.org/3/tutorial/venv.html
.. _Windows: https://www.youtube.com/watch?v=APOPm01BVrk
.. _MacOS: https://www.youtube.com/watch?v=Kg1Yvry_Ydk