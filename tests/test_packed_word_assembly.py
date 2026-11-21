"""Actual source register transport reaches native and Merlin tensor execution.

The masked byte shift is Triton f797708c test_core::test_inline_asm_packed.
The surrounding state/mixed-width examples and NumPy oracles are authored here.
Upstream license: docs/licenses/triton-LICENSE.txt.
"""

import json
import os

import numpy as np
import pytest
from test_packed_source_assembly import PROVIDER, TOOLS

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.frontend import capture_ttir, inspect_ttir
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import TensorABI, TensorBuffer, convert_shared

pytestmark = pytest.mark.skipif(not (TOOLS and PROVIDER), reason="existing source/native/Merlin tools required")


def state_program(width, pack, mode="independent"):
    per_register = min(32 // width, pack)
    registers = pack // per_register
    mask = sum(((1 << (width - 3)) - 1) << (slot * width) for slot in range(per_register))
    source = []
    for group in range(registers):
        a, b = 2 * registers + group, 3 * registers + group
        if mode == "coupled":
            source.append(f"shl.b32 ${group}, ${a}, 3;")
        else:
            source.extend((f"and.b32 ${group}, ${a}, {mask};", f"shl.b32 ${group}, ${group}, 3;"))
        source.append(f"xor.b32 ${registers + group}, ${a}, ${b};")
    if mode == "asymmetric":
        source.append("xor.b32 $0, $0, 1;")
    if mode == "hardware":
        source.append("mov.u32 $0, %clock;")
    return "\n".join(source), ",".join(["=&r"] * (2 * registers) + ["r"] * (2 * registers))


def cast_program(input_width, output_width, pack=4):
    inputs = max(32 // input_width, 1)
    outputs = max(32 // output_width, 1)
    output_registers = pack // outputs
    code = ["{ .reg .b32 %word;"]
    for register in range(output_registers):
        code.append(f"mov.b32 ${register}, 0;")
        for lane in range(outputs):
            slot = register * outputs + lane
            source = output_registers + slot // inputs
            code.extend((f"shr.u32 %word, ${source}, {(slot % inputs) * input_width};",
                         f"and.b32 %word, %word, {(1 << min(input_width, output_width)) - 1};",
                         f"shl.b32 %word, %word, {lane * output_width};",
                         f"or.b32 ${register}, ${register}, %word;"))
    code.append("}")
    return "\n".join(code), ",".join(["=&r"] * output_registers + ["r"] * (pack // inputs))


def capture(root, kind, profile="tensor", mode="independent"):
    from t2m.examples.packed_word_kernels import packed_word_cast, packed_word_state

    root.mkdir(parents=True)
    source = root / "source.ttir"
    if kind[0] == "state":
        _, width, pack = kind
        program, constraints = state_program(width, pack, mode)
        source.write_text(capture_ttir(packed_word_state,
            {name: f"*i{width}" if index < 4 else "i32" if index < 6 else "constexpr"
             for index, name in enumerate(packed_word_state.arg_names)},
            {"ASM": program, "CONSTRAINTS": constraints, "PACK": pack, "B": 8, "RANK2": width == 16}))
        spec = TensorABI("packed_word_state", tuple(TensorBuffer(name, ("n",), "read" if i < 2 else "write", f"i{width}")
            for i, name in enumerate(("x", "y", "out_a", "out_b"))), grid=(GridExtent("n", 8),),
            runtime_scalars=(RuntimeScalar("n", "i32", 4, bounds=(0, 67)),
                            RuntimeScalar("steps", "i32", 5, bounds=(-1, 3))))
    else:
        _, input_width, output_width = kind
        program, constraints = cast_program(input_width, output_width)
        source.write_text(capture_ttir(packed_word_cast,
            {"x": f"*i{input_width}", "out": f"*i{output_width}", "n": "i32", "ASM": "constexpr",
             "CONSTRAINTS": "constexpr", "PACK": "constexpr", "OUT_WIDTH": "constexpr", "B": "constexpr"},
            {"ASM": program, "CONSTRAINTS": constraints, "PACK": 4,
             "OUT_WIDTH": output_width, "B": 8}))
        spec = TensorABI("packed_word_cast", (TensorBuffer("x", ("n",), dtype=f"i{input_width}"),
            TensorBuffer("out", ("n",), "write", f"i{output_width}")), grid=(GridExtent("n", 8),),
            runtime_scalars=(RuntimeScalar("n", "i32", 2, bounds=(0, 67)),))
    inspect_ttir(source, root / "original-native")
    return convert_shared(source, spec, root / "translation", optimizer=os.environ["TRITON_SHARED_OPT"],
        memory_profile=profile, address_policy="source-words" if profile == "guarded-buffer" else "structured",
        assembly_provider="ptx-integer-v1")


def arguments(kind, n=13, steps=2):
    width = kind[1]
    unsigned, signed = getattr(np, f"uint{width}"), getattr(np, f"int{width}")
    words = np.asarray([0, 1, (1 << width) - 1, 1 << (width - 1), (1 << (width - 1)) - 1, 0x53], unsigned)
    x = np.resize(words, n).view(signed).copy()
    if kind[0] == "cast":
        wanted = x.view(unsigned).astype(getattr(np, f"uint{kind[2]}")).view(getattr(np, f"int{kind[2]}"))
        return [x, np.asarray(n, np.int32)], [wanted]
    y = np.resize(np.roll(words, 2), n).view(signed).copy()
    a, b = x.copy(), y.copy()
    for _ in range(steps):
        a, b = a << 3, a ^ b
    return [x, y, np.asarray(n, np.int32), np.asarray(steps, np.int32)], [a, b]


KINDS = [("state", 8, 4), ("state", 8, 8), ("state", 16, 2), ("state", 16, 4),
         ("cast", 8, 32), ("cast", 32, 8), ("cast", 8, 16)]
CASES = [(kind, profile) for kind in KINDS for profile in ("tensor", "buffer", "guarded-buffer")]


@pytest.fixture(scope="module", params=CASES)
def compiled(request, tmp_path_factory):
    from t2m.kernel_consumer import qualify_kernel
    from t2m.shared_bundle import write_shared_bundle

    kind, profile = request.param
    root = tmp_path_factory.mktemp("packed-words-" + "-".join(map(str, kind)) + "-" + profile)
    result = capture(root / "source", kind, profile)
    record, = [row for row in result.trace["transformations"] if row.get("kind") == "source-assembly-implementation"]
    assert record["packing_proof"]["method"] == "typed-word-bitvector-equivalence-v1"
    assert record["packing_proof"]["result"] == "unsat-counterexample"
    intent = json.loads(next(iter(result.module.ops)).attributes["prov.source_assembly_contracts"].data)[0]
    assert intent["register_packing"] == record["register_packing"]
    assert intent["provider_input_types"] == record["provider_input_types"]
    if kind[0] == "state":
        assert "scf.for" in result.mlir_text
    arrays, expected = arguments(kind)
    write_shared_bundle(result, root / "capture", arrays, expected, work=root / "native",
                        python=os.environ["T2M_NATIVE_PYTHON"])
    qualify_kernel(root / "capture", root / "merlin", merlin_root=os.environ["T2M_MERLIN_ROOT"],
        python=os.environ["T2M_MERLIN_PYTHON"], compiler_python=os.environ["T2M_NATIVE_PYTHON"],
        clang=os.environ["T2M_MERLIN_CLANG"])
    return root, kind


def test_complete_register_programs_reuse_one_library_for_original_and_empty_tail_domains(compiled):
    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    root, kind = compiled
    library = _record(root / "merlin/host/model.so")
    for index, (n, steps) in enumerate(((0, 3), (1, 0), (31, 3), (13, -1))):
        arrays, expected = arguments(kind, n, steps)
        inputs, reference = root / f"inputs-{index}.npz", root / f"reference-{index}.npz"
        np.savez(inputs, **{f"in{i}": value for i, value in enumerate(arrays)})
        np.savez(reference, **{f"out{i}": value for i, value in enumerate(expected)})
        invoke_kernel(root / "merlin", inputs, reference, root / f"invoke-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"], python=os.environ["T2M_MERLIN_PYTHON"])
        verify_invocation(root / f"invoke-{index}")
    assert _record(root / "merlin/host/model.so") == library


@pytest.mark.parametrize("mode", ("coupled", "asymmetric", "hardware", "padding"))
def test_no_physical_group_or_hardware_observation_is_invented(tmp_path, mode):
    with pytest.raises(ConversionError, match="grouping|register|hardware") as failure:
        capture(tmp_path / "source", ("state", 8, 2 if mode == "padding" else 4), mode=mode)
    assert failure.value.opcode == "tt.elementwise_inline_asm" and failure.value.source_id
    assert not (tmp_path / "source/translation/model.mlir").exists()


@pytest.mark.parametrize("damage,diagnostic", (("overflow", "overflow flags"),
    ("shift", "undefined"), ("semantic-attribute", "semantic property")))
def test_a_provider_cannot_supply_unproved_arithmetic_semantics(tmp_path, monkeypatch, damage, diagnostic):
    from xdsl.dialects import arith, builtin

    from t2m import source_assembly
    from t2m.source_assembly import AssemblyExpansion

    class Invalid:
        def lower(self, request, arguments):
            if damage == "shift":
                op = arith.ShLIOp(arguments[0], arguments[1])
                operations = [op]
            else:
                one = arith.ConstantOp.from_int_and_width(1, builtin.i32)
                op = arith.AddiOp(arguments[0], one.result,
                    overflow=arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW]) if damage == "overflow"
                    else arith.IntegerOverflowAttr([]))
                if damage == "semantic-attribute":
                    op.attributes["assumed_numeric_policy"] = builtin.UnitAttr()
                operations = [one, op]
            return AssemblyExpansion(operations, [op.result, arguments[1]], {})

    monkeypatch.setattr(source_assembly, "load_provider", lambda name: (Invalid(), {}))
    with pytest.raises(ConversionError, match=diagnostic) as failure:
        capture(tmp_path / "source", ("state", 8, 4))
    assert failure.value.opcode == "tt.elementwise_inline_asm" and failure.value.source_id
