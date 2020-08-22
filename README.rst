=======
flowsym
=======

.. image:: https://badge.fury.io/py/flowsym.svg
    :target: https://badge.fury.io/py/flowsym

.. image:: https://travis-ci.com/harmslab/flowsym.svg?branch=master
    :target: https://travis-ci.com/harmslab/flowsym

.. image:: https://readthedocs.org/projects/flowsym/badge/?version=latest
    :target: https://flowsym.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



**A Python API for simulating flow cytometry data.**


* Free software: MIT license
* Documentation: https://flowsym.readthedocs.io/


Installation
--------
1. Clone and navigate to this directory

.. code-block:: console

    $ git clone git://github.com/harmslab/flowsym
    $ cd flowsym
    

2. Create a new virtual environment using requirements on `environment.yml` file.

.. code-block:: console

    $ conda env create -f flowsym/environment. yml

3. Alternatively, install the requirements manually. Note that `hdbscan` must be installed using conda-forge channel

.. code-block:: console

    $ conda install -c conda-forge hdbscan
    $ pip install -r requirements.txt


Examples
-------
Coming soon.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
