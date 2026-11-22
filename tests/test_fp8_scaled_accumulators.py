"""Original FP8 C, wide operands and rounding at every source scaled dot."""

import math
import os
from fractions import Fraction

import numpy as np
import pytest
from test_narrow_atomics import REQUIRED, invoke

from t2m.bundle import _record
from t2m.dynamic_abi import GridExtent
from t2m.examples import fp8_arithmetic_reference as oracle
from t2m.examples.fp8_reference import FORMATS, decode
from t2m.frontend import inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not REQUIRED, reason="existing pinned/native/Merlin tools required")


def program(dtype):
    names = ("a", "b", "c", "out", "original")
    dtypes = ("f16", "f16", dtype, dtype, dtype)
    buffers = tuple(TensorBuffer(name, (2, 2) if i < 2 else ("n", 4),
        "write" if i >= 3 else "read", dtypes[i]) for i, name in enumerate(names))
    abi = TensorABI("scaled", buffers, grid=(GridExtent("n", 1),), runtime_scalars=(
        RuntimeScalar("n", "i32", 5, bounds=(0, 64)), RuntimeScalar("steps", "i32", 6, bounds=(0, 3)),
        RuntimeScalar("enabled", "i1", 7)))
    args = ",".join(f"%{name}:!tt.ptr<{type_}>" for name, type_ in zip(names, dtypes, strict=True))
    out = f"tensor<2x2x{dtype}>"
    lines = [f"module {{tt.func public @scaled({args},%n:i32,%steps:i32,%enabled:i1) {{", '''
      %offset=arith.constant dense<[[0,1],[2,3]]>:tensor<2x2xi32>
      %pid=tt.get_program_id x:i32
      %four=arith.constant 4:i32
      %base=arith.muli %pid,%four:i32
      %bases=tt.splat %base:i32->tensor<2x2xi32>
      %index=arith.addi %offset,%bases:tensor<2x2xi32>
      %live=arith.cmpi slt,%pid,%n:i32
      %mask=tt.splat %live:i1->tensor<2x2xi1>
      %zero32=arith.constant 0:i32
      %one32=arith.constant 1:i32
    ''', f"%zero=arith.constant dense<0.0>:{out}"]
    for name, type_ in zip(names, dtypes, strict=True):
        ptr = f"tensor<2x2x!tt.ptr<{type_}>>"
        lines += [f"%p_{name}=tt.splat %{name}:!tt.ptr<{type_}>->{ptr}",
            f"%q_{name}=tt.addptr %p_{name},%{'offset' if name in ('a', 'b') else 'index'}:{ptr},tensor<2x2xi32>"]
    for name in ("a", "b"):
        lines.append(f"%v_{name}=tt.load %q_{name}:tensor<2x2x!tt.ptr<f16>>")
    lines += [f"%v_c=tt.load %q_c,%mask,%zero:tensor<2x2x!tt.ptr<{dtype}>>",
        f"%state=scf.for %i=%zero32 to %steps step %one32 iter_args(%carry=%v_c)->({out}):i32 {{",
        f"%next=scf.if %enabled -> ({out}) {{"]
    attrs = "a_elem_type=6:i32,b_elem_type=6:i32,fastMath=false,lhs_k_pack=true,rhs_k_pack=true,operandSegmentSizes=array<i32:1,1,1,0,0>"
    for result, a, b, c in (("first", "a", "b", "carry"), ("second", "b", "a", "first")):
        lines.append(f'%{result}="tt.dot_scaled"(%v_{a},%v_{b},%{c}) {{{attrs}}}:'
            f"(tensor<2x2xf16>,tensor<2x2xf16>,{out})->{out}")
    lines += [f"scf.yield %second:{out}", f"}} else {{scf.yield %carry:{out}}}",
        f"scf.yield %next:{out}", "}",
        f"tt.store %q_out,%state,%mask:tensor<2x2x!tt.ptr<{dtype}>>",
        f"tt.store %q_original,%v_c,%mask:tensor<2x2x!tt.ptr<{dtype}>>", "tt.return", "}}"]
    return "\n".join(lines), abi


def arguments(dtype, n, steps, enabled, shift=0):
    a = np.asarray([[1.0625, 0.03125], [-1.9375, 0.5]], np.float16)
    b = np.asarray([[1.0625, -0.25], [0.125, 1.1875]], np.float16)
    if shift:
        a = np.flip(a, 0).copy()
        b = -b
    c = ((np.arange(n * 4, dtype=np.uint32) + shift) & 255).astype(np.uint8).reshape(n, 4)
    return [a, b, c, np.asarray(n, np.int32), np.asarray(steps, np.int32), np.asarray(enabled)]


def reference(arrays, dtype, *, narrow_operands=False, round_each=True):
    a, b, c, _, steps, enabled = arrays
    matrices = [[Fraction(float(v)) for v in matrix.flat] for matrix in (a, b)]
    if narrow_operands:
        matrices = [[oracle.value(oracle.rounded(v, dtype), dtype) for v in matrix] for matrix in matrices]
    products = []
    for first, second in (matrices, matrices[::-1]):
        products.append([sum(first[r * 2 + k] * second[k * 2 + col] for k in range(2))
                         for r in range(2) for col in range(2)])
    output = []
    for row in c:
        state = [int(v) for v in row]
        if not round_each:
            values = [oracle.value(v, dtype) for v in state]
        for _ in range(int(steps) if bool(enabled) else 0):
            for product in products:
                if round_each:
                    next_ = []
                    for old, delta in zip(state, product, strict=True):
                        number = oracle.value(old, dtype)
                        next_.append(old if not math.isfinite(number) else oracle.rounded(number + delta, dtype))
                    state = next_
                else:
                    values = [v + delta for v, delta in zip(values, product, strict=True)]
        if not round_each:
            state = [oracle.nan(dtype) if math.isnan(v) else oracle.infinity(v < 0, dtype)
                     if math.isinf(v) else oracle.rounded(v, dtype) for v in values]
        output.append(state)
    return np.asarray(output, np.uint8).reshape(c.shape)


@pytest.mark.parametrize("dtype", tuple(FORMATS))
def test_fp8_c_rounds_at_each_dot_and_keeps_wide_input_products(tmp_path, dtype):
    from merlin.api import StagedProgram, compile
    from merlin.compiler.frontend.inputs import stage_compile_inputs
    from test_fp8_arithmetic import compare

    from t2m.merlin_backend import MerlinHostBackend
    from t2m.shared_bundle import write_shared_bundle

    source, abi = program(dtype)
    path = tmp_path / "source.ttir"
    path.write_text(source)
    inspect_ttir(path, tmp_path / "source-verification")
    result = convert_shared(path, abi, tmp_path / "translation", memory_profile="guarded-buffer", address_policy="source-words")
    dots = [row for row in result.trace["transformations"] if row["kind"] == "source-scaled-dot"]
    assert len(dots) == 2 and all(row["result_type"] == dtype and row["intermediate_type"] == "f64" for row in dots)
    arrays = arguments(dtype, 64, 3, True)
    expected = reference(arrays, dtype)
    finite = np.asarray([not math.isnan(decode(int(v), dtype)) for v in expected.flat]).reshape(expected.shape)
    assert np.any((expected != reference(arrays, dtype, narrow_operands=True)) & finite)
    assert np.any((expected != reference(arrays, dtype, round_each=False)) & finite)
    write_shared_bundle(result, tmp_path / "capture", arrays, [expected, arrays[2]], work=tmp_path / "native",
        python=os.environ["T2M_NATIVE_PYTHON"])
    stage_compile_inputs(tmp_path / "capture", tmp_path / "stage")
    product = compile(StagedProgram(tmp_path / "stage"), backend=MerlinHostBackend(
        merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"],
        compiler_python=os.environ["T2M_NATIVE_PYTHON"], clang=os.environ["T2M_MERLIN_CLANG"]), output=tmp_path / "compiled")
    library = _record(tmp_path / "compiled/host/model.so")
    for index, (n, steps, enabled, shift) in enumerate(((64, 3, True, 0), (64, 2, True, 17),
            (13, 0, True, 3), (1, 3, False, 128), (0, 3, True, 0))):
        arrays = arguments(dtype, n, steps, enabled, shift)
        before = [a.tobytes() for a in arrays]
        actual = invoke(tmp_path, arrays, f"invoke-{index}")
        assert len(actual) == 2
        compare(actual[0], reference(arrays, dtype), dtype)
        compare(actual[1], arrays[2], dtype, exact=True)
        assert before == [a.tobytes() for a in arrays]
    assert _record(tmp_path / "compiled/host/model.so") == library
    assert not any(p.name in {"inputs.npz", "golden.npy", "extra.npz"}
        for p in (tmp_path / "compiled/frontend").rglob("*"))
    product.verify()
