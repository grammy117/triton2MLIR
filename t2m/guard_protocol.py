"""Semantic ownership encodings for ordinary and atomic logical memory.

These are compiler-owned verification tags, not target opcodes or source
atomic implementations. Complemented fragments make foreign participants
detectable under atomic OR, including all logical program-ID bits.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OwnershipProtocol:
    name: str
    fragment_bits: int
    writer_bit: int
    ordinary_bit: int = 0
    atomic_bit: int = 0

    @property
    def fragment_mask(self):
        return (1 << self.fragment_bits) - 1

    @property
    def tag_mask(self):
        return (1 << (2 * self.fragment_bits)) - 1


ORDINARY = OwnershipProtocol("sticky-complemented-program-fragments", 31, 1 << 62)
ATOMIC = OwnershipProtocol("sticky-complemented-atomic-participants-v1", 30, 1 << 60, 1 << 61, 1 << 62)
