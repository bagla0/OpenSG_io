"""Sphinx config for the OpenSG_io documentation site (GitHub Pages)."""
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "OpenSG_io"
author = "Akshat Bagla"
copyright = "2026, Akshat Bagla (bagla0). Built with Claude."
release = "0.3.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "OpenSG_io"
html_theme_options = {
    "source_repository": "https://github.com/bagla0/OpenSG_io",
    "source_branch": "main",
    "source_directory": "docs/",
}
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# autodoc should not hard-fail if optional deps (windIO, jax) are absent on the docs builder
autodoc_mock_imports = ["windIO", "jax", "dolfinx", "pypardiso"]
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
