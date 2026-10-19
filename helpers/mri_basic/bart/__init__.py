"""Wrapper around the BART command-line tool: exposes ``bart`` and
``clear_stdouterr`` (CFL file I/O lives in cfl.py, WSL paths in wslsupport.py).
"""

from .bart import bart, clear_stdouterr
