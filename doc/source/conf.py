# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'vampire-analysis'
copyright = f'{date.today().year}, Teng-Jui Lin'
author = 'Teng-Jui Lin'

# The full version, including alpha/beta/rc tags
release = '0.2.0.dev1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.napoleon',  # use numpydoc instead
    'myst_parser',  # parse markdown
    'numpydoc',  # format autodoc like numpy
    'nbsphinx',  # parse Jupyter Notebook
    # 'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    # 'autodoc2',  # allow markdown docstring, but competes with numpydoc
]

autosummary_generate = True
# numpydoc interferes with autosummary to generate two method sections
# for each class.
# https://stackoverflow.com/questions/34216659/sphinx-autosummary-produces-two-summaries-for-each-class
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints', '.DS_Store']


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "github_url": "https://github.com/tengjuilin/vampire-analysis",
    'logo': {
        'image_light': 'vampire-logo.png',
        'image_dark': 'vampire-logo.png',
    }
}
html_favicon = "_static/vampire-logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    "vampire.css",
]

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/dev', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# -----------------------------------------------------------------------------
# myst-parser configuration
# -----------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    'dollarmath',
    'amsmath',
    # 'linkify',
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -----------------------------------------------------------------------------
# autodoc2 configuration
# -----------------------------------------------------------------------------
# autodoc2_packages = ["../../vampire"]
# autodoc2_render_plugin = "myst"
# autodoc2.config.PackageConfig.auto_mode = True
