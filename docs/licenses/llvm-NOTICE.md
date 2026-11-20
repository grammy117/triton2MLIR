# LLVM/MLIR integer range inference

The signed-remainder inference in `t2m/integer_intervals.py` adapts the
sign/modulus and contiguous constant-divisor strategy from
`mlir/lib/Interfaces/Utils/InferIntRangeCommon.cpp`, `inferRemS`, at LLVM
revision `a47bddccec30255619bb8c37fa59700e661d4e66`. The Python implementation
adds source-width and native poison admission checks and tightens bounds by
the dividend interval. Existing signed-division inference uses the same
upstream endpoint/sign-partition strategy. This is modified range inference,
not a copy of the native solver or its result representation.

The upstream source notice is:

```text
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

The full license is retained in [llvm-LICENSE.txt](llvm-LICENSE.txt).
Program-point and loop-interface analysis references and inspected upstream
tests are recorded in [source_scope_ranges.md](../source_scope_ranges.md).
