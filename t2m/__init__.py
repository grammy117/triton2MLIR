"""Triton to portable, provenance-bearing tensor MLIR."""

from .api import convert
from .spec import BufferSpec, KernelSpec

__all__ = ["BufferSpec", "KernelSpec", "convert"]
__version__ = "0.1.0"

