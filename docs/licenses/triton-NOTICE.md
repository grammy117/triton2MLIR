# Triton inline assembly register packing

`t2m/packed_words.py` adapts the register packing order and result unpacking
algorithm from `ElementwiseInlineAsmOpConversion` in Triton's
`lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp` at
`f797708c0626e5f9840ca5b0a98790e2c7cb09ad`.

Copyright 2018-2020 Philippe Tillet. Copyright 2020-2022 OpenAI.
The upstream MIT permission and disclaimer are retained in
`triton-LICENSE.txt`. Upstream numerical fixtures are referenced by revision
in `docs/source_assembly.md`; no CUDA build or device runtime is bundled.
