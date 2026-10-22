"""Sphinx configuration for the vampire-analysis documentation.

Build with::

    uv run sphinx-build -W --keep-going -b html docs docs/_build/html

Executable pages (MyST markdown with ``file_format: mystnb`` front matter) are
run at build time; any exception fails the build, so they double as tests.
Set ``NB_EXECUTION_MODE=cache`` to reuse outputs between local rebuilds.
"""

import os
from datetime import date
from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------

project = "vampire-analysis"
author = "Teng-Jui Lin"
copyright = f"2021-{date.today().year}, {author}"
release = _version("vampire-analysis")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",  # also registers myst_parser for plain .md pages
    "sphinx_copybutton",
    "sphinx_llms_txt",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
root_doc = "index"
default_role = "py:obj"

# -- autodoc / autosummary / numpydoc ---------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "none"  # types are spelled out in the numpydoc docstrings
autodoc_default_options = {"member-order": "bysource"}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "DataFrame": "pandas.DataFrame",
    "Series": "pandas.Series",
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    "Colormap": "matplotlib.colors.Colormap",
    "Pipeline": "sklearn.pipeline.Pipeline",
    "Generator": "numpy.random.Generator",
    "RandomState": "numpy.random.RandomState",
    "ndarray": "numpy.ndarray",
    "array-like": ":term:`numpy:array_like`",
    "path-like": ":term:`python:path-like object`",
}
numpydoc_xref_ignore = "all"  # link only the aliases above (+ numpydoc defaults)

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
}

# -- MyST / myst-nb ----------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3
myst_substitutions = {"version": release}

nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "force")
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_timeout = 300
nb_merge_streams = True
nb_execution_excludepatterns = []

# -- sphinx-copybutton ---------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- llms.txt ----------------------------------------------------------------

llms_txt_title = "vampire-analysis"
llms_txt_summary = (
    "vampire-analysis (import vampire as vp) clusters object contours from "
    "segmented microscopy images into shape modes (VAMPIRE, Phillip et al., "
    "Nat Protoc 2021) with a scikit-learn-style API. Start with for_agents "
    "for a compact API and invariants summary."
)

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"vampire-analysis {release}"
html_static_path = ["_static"]
html_css_files = ["vampire.css"]
html_favicon = "_static/vampire-logo.png"
html_baseurl = "https://vampire.readthedocs.io/en/latest/"
html_theme_options = {
    "logo": {
        "image_light": "_static/vampire-logo.png",
        "image_dark": "_static/vampire-logo.png",
        "text": "vampire-analysis",
    },
    "github_url": "https://github.com/tengjuilin/vampire-analysis",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/vampire-analysis/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {
    "github_user": "tengjuilin",
    "github_repo": "vampire-analysis",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}
html_sidebars = {"index": [], "changelog": [], "license": [], "references": []}
