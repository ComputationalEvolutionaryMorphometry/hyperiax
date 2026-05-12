"""hyperiax.io — optional I/O backends.

This subpackage is L2: it may import the L1 ``hyperiax.core`` types and
external libraries (``ete3``). Importing ``hyperiax.io`` itself does not
pull ``ete3``; only the function bodies do (so that a user without the
``[io]`` extra can still ``import hyperiax``).
"""

from . import newick

__all__ = ["newick"]
