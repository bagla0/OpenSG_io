"""Sphinx config for the OpenSG_io documentation site (GitHub Pages, pydata-sphinx-theme)."""
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "OpenSG_io"
author = "Akshat Bagla"
copyright = "2026, Akshat Bagla (bagla0)"
release = "0.3.0"

extensions = [
    "myst_nb",                       # MyST markdown + Jupyter notebooks (.ipynb)
    "sphinx_design",                 # grids, cards, tabs, buttons on the landing page
    "sphinx_copybutton",             # copy button on code blocks
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "attrs_inline", "substitution"]
nb_execution_mode = "off"            # notebooks are committed pre-executed -> render stored outputs
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---- HTML / theme ----
html_theme = "pydata_sphinx_theme"
html_title = "OpenSG_io"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_show_sourcelink = False
html_context = {
    "github_user": "bagla0",
    "github_repo": "OpenSG_io",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}
html_theme_options = {
    "logo": {"text": "OpenSG_io"},
    "github_url": "https://github.com/bagla0/OpenSG_io",
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/bagla0/OpenSG_io", "icon": "fa-brands fa-github"},
        {"name": "OpenSG-TW", "url": "https://github.com/bagla0/OpenSG-TW", "icon": "fa-solid fa-cube"},
    ],
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "show_prev_next": True,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "github-dark",
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "announcement": "OpenSG_io &mdash; prepare OpenSG cross-section inputs from windIO, PreVABS, or OpenFAST.",
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}
html_sidebars = {"index": []}        # full-width landing page (no left sidebar on the home page)

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
autodoc_mock_imports = ["windIO", "jax", "dolfinx", "pypardiso", "matplotlib"]
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
