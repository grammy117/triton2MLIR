"""PTX source programs execute as standard MLIR through actual Merlin libraries."""

import json
import os

import numpy as np
import pytest
from t2m_ptx import Provider
from xdsl.dialects import builtin
from xdsl.ir import Block

from t2m.dynamic_abi import GridExtent
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared
from t2m.source_assembly import AssemblyRefusal, AssemblyRequest

OPT, NATIVE = os.environ.get("TRITON_SHARED_OPT"), os.environ.get("T2M_NATIVE_PYTHON")
MERLIN = all(os.environ.get(n) for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))


def program(kind, width):
    if kind == "arithmetic":
        return f'''{{
          .reg .b{width} %r<2>;
          .reg .pred %p;
          mul.lo.u{width} %r0, $2, $3;
          add.u{width} %r1, %r0, -1;
          setp.lt.s{width} %p, $2, $3;
          selp.u{width} $0, %r1, $3, %p;
          xor.b{width} $1, $2, $3;
          @!%p add.u{width} $1, $1, 1;
        }}'''
    if kind == "shifts":
        return f"shl.b{width} $0, $2, $4; shr.s{width} $1, $3, $4;"
    if kind == "minmax":
        return f'''{{ .reg .pred %p;
          min.s{width} $0, $2, $3;
          max.u{width} $1, $2, $3;
          setp.ne.u{width} %p, $2, 0;
          @%p neg.s{width} $0, $0;
        }}'''
    if kind == "word-math":
        return f'''{{ .reg .b{width} %r0, %r1, %r2, %r3, %r4;
          // Source order and every modular word remain explicit.
          sub.u{width} %r0, $2, $3;
          mad.lo.s{width} %r1, %r0, $3, 0x1234;
          abs.s{width} %r2, %r1;
          neg.s{width} %r3, %r2;
          mov.b{width} %r4, %r3;
          max.s{width} $0, %r4, $2;
          min.u{width} $1, %r1, $3;
        }}'''
    if kind == "logical-shifts":
        return f"shr.u{width} $0, $2, $4; shr.b{width} $1, $3, $4;"
    if kind == "predicates":
        return f'''{{ .reg .pred %p<6>;
          setp.eq.b{width} %p0, $2, $3;
          setp.ne.b{width} %p1, $2, 0;
          setp.le.u{width} %p2, $2, $3;
          setp.gt.s{width} %p3, $2, $3;
          setp.ge.s{width} %p4, $2, $3;
          setp.lt.u{width} %p5, $2, $3;
          not.pred %p5, %p5;
          and.pred %p0, %p0, %p1;
          or.pred %p2, %p2, %p3;
          xor.pred %p4, %p4, %p5;
          selp.b{width} $0, $2, $3, %p0;
          @%p2 sub.u{width} $0, $0, 3;
          selp.s{width} $1, $2, $3, !%p4;
        }}'''
    return f'''{{ .reg .b{width} %r;
       not.b{width} %r, $3;
       and.b{width} $0, $2, %r;
       or.b{width} $1, $2, $3;
    }}'''


def capture(root, kind, width, profile="guarded-buffer"):
    from t2m.examples.assembly_kernels import assembly_word_state
    from t2m.frontend import capture_ttir

    source = root / "source.ttir"
    names = ("x", "y", "shifts", "out_a", "out_b")
    source.write_text(capture_ttir(assembly_word_state, {
        **{name: "*i32" if name == "shifts" else "*i" + str(width) for name in names},
        "n": "i32", "iterations": "i32", "ASM": "constexpr", "B": "constexpr", "WORDS64": "constexpr"},
        {"ASM": program(kind, width), "B": 32, "WORDS64": width == 64}))
    abi = TensorABI("assembly_word_state", tuple(TensorBuffer(name, ("n",),
        "write" if name.startswith("out") else "read", "i32" if name == "shifts" else "i" + str(width)) for name in names),
        grid=(GridExtent("n", 32),), runtime_scalars=(RuntimeScalar("n", "i32", 5, bounds=(0, 67)),
                                                    RuntimeScalar("iterations", "i32", 6, bounds=(0, 4))))
    return source, abi, convert_shared(source, abi, root / "translation", optimizer=OPT,
        memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured",
        assembly_provider="ptx-integer-v1")


def arguments(n, iterations, kind, width):
    dtype = np.int64 if width == 64 else np.int32
    modulus, mask = 1 << width, (1 << width) - 1

    def signed(word):
        word &= mask
        return word - modulus if word >= modulus // 2 else word

    x = np.resize(np.asarray([0, -1, 1, -7, modulus // 2 - 1, -modulus // 2, modulus // 4 + 3], dtype), n)
    y = np.resize(np.asarray([-1, 0, -modulus // 2, modulus // 2 - 1, 2, -3, 7, -11], dtype), n)
    shifts = np.resize(np.asarray([0, 1, width - 1, width, width + 1, -1, -modulus // 2 if width == 32 else -(1 << 31)], np.int32), n)
    a, b = list(map(int, x)), list(map(int, y))
    for _ in range(iterations):
        first, second = [], []
        for left, right, count in zip(a, b, shifts, strict=True):
            unsigned_left, unsigned_right = left & mask, right & mask
            if kind == "arithmetic":
                low = (unsigned_left * unsigned_right - 1) & mask
                p = left < right
                c, d = (low if p else unsigned_right), ((unsigned_left ^ unsigned_right) + (not p))
            elif kind == "shifts":
                count = min(int(count) & ((1 << 32) - 1), width)
                c, d = unsigned_left << count, right >> count
            elif kind == "minmax":
                c, d = (-min(left, right) if left else min(left, right)), max(unsigned_left, unsigned_right)
            elif kind == "word-math":
                product = signed((left - right) * right + 0x1234)
                absolute = signed(abs(product))
                c, d = max(signed(-absolute), left), min(product & mask, unsigned_right)
            elif kind == "logical-shifts":
                count = min(int(count) & ((1 << 32) - 1), width)
                c, d = unsigned_left >> count, unsigned_right >> count
            elif kind == "predicates":
                p0 = left == right and left != 0
                p2 = unsigned_left <= unsigned_right or left > right
                p4 = (left >= right) != (unsigned_left >= unsigned_right)
                c, d = (left if p0 else right) - (3 if p2 else 0), (left if not p4 else right)
            else:
                c, d = unsigned_left & ~unsigned_right, unsigned_left | unsigned_right
            first.append(signed(c))
            second.append(signed(d))
        a, b = first, second
    return [x, y, shifts, np.asarray(n, np.int32), np.asarray(iterations, np.int32)], [np.asarray(a, dtype), np.asarray(b, dtype)]


@pytest.mark.skipif(not (OPT and NATIVE and MERLIN), reason="existing shared/native/Merlin tools required")
@pytest.mark.parametrize("kind,width,profile", [(k, w, "guarded-buffer") for k in ("arithmetic", "shifts", "minmax", "logic", "word-math", "logical-shifts", "predicates") for w in (32, 64)]
                         + [("arithmetic", 32, p) for p in ("tensor", "buffer")])
def test_actual_libraries_retain_complete_integer_programs_and_runtime_state(tmp_path, kind, width, profile):
    from t2m.invocation import invoke_kernel
    from t2m.kernel_consumer import qualify_kernel, verify_kernel_qualification
    from t2m.shared_bundle import write_shared_bundle

    _, _, result = capture(tmp_path, kind, width, profile)
    report, = [t for t in result.trace["transformations"] if t["kind"] == "source-assembly-implementation"]
    assert report["provider"]["distribution"] == "triton2mlir-ptx-semantics" and report["contract"]["instructions"]
    assert not any(op.name.startswith(("tt.", "llvm.", "nvvm.", "gpu.")) for op in result.module.walk())
    arrays, expected = arguments(37, 3, kind, width)
    bundle, compiled = tmp_path / "capture", tmp_path / "merlin"
    write_shared_bundle(result, bundle, arrays, expected, work=tmp_path / "native", python=NATIVE)
    assert any("optional source-assembly providers" in d for d in json.loads((bundle / "compatibility.json").read_text())["differences"])
    root, python, clang = (os.environ[n] for n in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG"))
    qualify_kernel(bundle, compiled, merlin_root=root, python=python, compiler_python=NATIVE, clang=clang)
    receipt = verify_kernel_qualification(compiled)
    for index, (n, count) in enumerate(((0, 0), (1, 4), (32, 0), (67, 2))):
        arrays, expected = arguments(n, count, kind, width)
        inputs, reference = tmp_path / "inputs.npz", tmp_path / "reference.npz"
        np.savez(inputs, **{f"in{j}": value for j, value in enumerate(arrays)})
        np.savez(reference, **{f"out{j}": value for j, value in enumerate(expected)})
        invocation = invoke_kernel(compiled, inputs, reference, tmp_path / f"invoke{index}", merlin_root=root, python=python)
        assert invocation["evaluation"]["functional_inputs_preserved"]
    assert verify_kernel_qualification(compiled) == receipt


@pytest.mark.parametrize("assembly", [
    "mov.u32 $0, %clock;", "ld.global.u32 $0, [$1];", "add.sat.s32 $0, $1, 1;",
    "add.cc.u32 $0, $1, 1;", "div.u32 $0, $1, 0;", "cvt.u32.u64 $0, $1;",
    "add.u32 $0, %undeclared, 1;", "add.u32 $1, $1, 1;", "mov.u32 $0, $0;",
    "mov.u32 $0, 1", "/* unterminated", "{ mov.u32 $0, 1;", "{ { mov.u32 $0, 1; } }",
    ".reg .u32 %r; .reg .u32 %r; mov.u32 $0, 1;", "mov.u32 $0, 1; trailing garbage;",
    ".reg .pred %p; setp.ne.u32 %p, $1, 0; @%p mov.u32 $0, 1;",
    ".reg .u32 %; mov.u32 $0, 1;", ".reg .u32 %tid.x; mov.u32 $0, 1;",
    ".reg .u32 %r<999999999999999999999>; mov.u32 $0, 1;", "mov.u32 $0, ١;",
])
def test_unknown_effects_state_flags_and_uninitialized_values_are_refused(assembly):
    block = Block(arg_types=[builtin.i32])
    request = AssemblyRequest(assembly, "=r,r", (builtin.i32,), (builtin.i32,), 1)
    with pytest.raises(AssemblyRefusal):
        Provider().lower(request, list(block.args))
