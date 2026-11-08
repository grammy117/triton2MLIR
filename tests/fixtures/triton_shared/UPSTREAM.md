# Upstream fixture provenance

- Repository: https://github.com/facebookincubator/triton-shared
- Revision: `17f0c234ce474bee334af36e351c62eca7852836`
- Triton source pin: `142586f90aac9b917af1081e4697b085027e13b2`
- Original path: `test/Conversion/TritonToStructured/addptr_2d_example.mlir`
- License: MIT; the upstream license is retained in `LICENSE`.

The input module and its explanatory comments are retained. Original RUN/CHECK directives are
removed and an attribution header is added. Expected compiler output is not manufactured from
FileCheck comments. `test_upstream.py` runs this fixture against an actual configured optimizer
for structured, tensor-compute and full middle-layer stages; those checks skip without that tool.

Passing a stage test establishes upstream execution and the observed stage representation. It does
not establish triton2MLIR's functional tensor ABI, source correspondence or numerical equivalence.
