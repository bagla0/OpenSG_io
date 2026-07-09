"""Sphinx configuration for the OpenSG_io documentation site.

Same stack and look as the OpenSG-TW documentation (sphinx-rtd-theme + myst-nb,
matching the upstream OpenSG docs at wenbinyugroup.github.io/OpenSG), deployed to
GitHub Pages.  Tutorial notebooks are committed pre-executed and rendered as-is.
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # repo root, so autodoc can import opensg_io.*

project = "OpenSG_io"
author = "Akshat Bagla (bagla0)"
copyright = "%d, Akshat Bagla" % datetime.date.today().year
release = "0.3.0"

# ---------------------------------------------------------------- extensions
extensions = [
    "myst_nb",            # MyST markdown + executed Jupyter notebooks (.ipynb)
    "sphinx_design",      # grids / cards on the tutorials index
    "sphinx_copybutton",  # copy button on code blocks
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True

myst_enable_extensions = [
    "colon_fence", "deflist", "dollarmath", "amsmath", "attrs_inline", "substitution",
]
myst_dmath_double_inline = True

# Notebooks are committed already executed -> just render the stored outputs.
nb_execution_mode = "off"
nb_merge_streams = True

source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute"]

# ---------------------------------------------------------------- HTML / theme
# Read-the-Docs theme, identical to the OpenSG-TW documentation (and the upstream
# OpenSG docs), of which OpenSG_io is the input-preparation companion.
html_theme = "sphinx_rtd_theme"
html_title = "OpenSG_io"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_last_updated_fmt = "%Y-%m-%d"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None),
                       "numpy": ("https://numpy.org/doc/stable", None)}
# autodoc imports opensg_io; mock everything heavier than numpy/yaml
autodoc_mock_imports = ["windIO", "jax", "dolfinx", "pypardiso", "matplotlib", "pyvista",
                        "scipy", "gmsh"]
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
suppress_warnings = ["docutils", "myst.substitution", "autodoc"]
