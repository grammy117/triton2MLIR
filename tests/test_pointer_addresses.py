"""Closed address arithmetic keeps logical memory and full source-width words."""

import json
import os

import numpy as np
import pytest

from t2m.diagnostics import ConversionError
from t2m.dynamic_abi import GridExtent
from t2m.pointer_addresses import normalize_pointer_addresses
from t2m.scalar_abi import RuntimeScalar
from t2m.shared_tensor import (
    TensorABI,
    TensorBuffer,
    _source_ids,
    convert_shared,
    parse_shared,
)

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")
REQUIRED = (
    OPT
    and NATIVE
    and all(
        os.environ.get(k)
        for k in ("T2M_MERLIN_ROOT", "T2M_MERLIN_PYTHON", "T2M_MERLIN_CLANG")
    )
)


def capture(root, width):
    from t2m.examples.pointer_address_kernels import displaced_words
    from t2m.frontend import capture_ttir

    scalars = (
        RuntimeScalar("n", "i32", 3, bounds=(0, 31)),
        RuntimeScalar("steps", "i32", 4, bounds=(0, 4)),
        RuntimeScalar("origin", "i32", 5, bounds=(-8192, 8192)),
        RuntimeScalar("stride", "i32", 6, bounds=(-24, 24)),
        RuntimeScalar("capacity", "i32", 7, bounds=(0, 128)),
    )
    buffers = tuple(
        TensorBuffer(
            name, ("capacity",), "readwrite" if name == "out" else "read", f"i{width}"
        )
        for name in ("x", "y", "out")
    )
    source = root / "source.ttir"
    source.write_text(
        capture_ttir(
            displaced_words,
            {
                **{b.name: f"*{b.dtype}" for b in buffers},
                **{s.name: s.dtype for s in scalars},
                "B": "constexpr",
                "WIDTH": "constexpr",
            },
            {"B": 8, "WIDTH": width},
        )
    )
    abi = TensorABI(
        displaced_words.__name__,
        buffers,
        grid=(GridExtent("n", 8),),
        runtime_scalars=scalars,
    )
    return convert_shared(
        source,
        abi,
        root / "translation",
        optimizer=OPT,
        memory_profile="guarded-buffer",
        address_policy="source-words",
    )


def arguments(width, n=13, steps=3, origin=8, stride=24, capacity=128):
    unsigned = np.dtype(f"uint{width}")
    signed = np.dtype(f"int{width}")
    pattern = np.asarray(
        [
            0,
            1,
            (1 << width) - 1,
            1 << (width - 1),
            (1 << (width - 1)) - 1,
            (1 << 53) + 1 if width == 64 else (1 << min(24, width - 2)) + 1,
        ],
        dtype=unsigned,
    )
    x = np.resize(pattern, capacity).view(signed).copy()
    y = np.resize(pattern[::-1], capacity).view(signed).copy()
    initial = np.resize(np.roll(pattern, 2), capacity).view(signed).copy()
    expected = initial.view(unsigned).copy()
    for row in range(steps):
        for lane in range(n):
            index = origin + row * stride + lane
            assert 0 <= index < capacity
            word = (
                int((x if lane % 2 == 0 else y).view(unsigned)[index])
                if lane % 3 != 1
                else ((1 << width) - 17)
            )
            expected[index] = word ^ (row + 1)
    return [
        x,
        y,
        initial,
        *(np.asarray(v, np.int32) for v in (n, steps, origin, stride, capacity)),
    ], expected.view(signed)


@pytest.mark.skipif(not REQUIRED, reason="existing native/Merlin tools required")
@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_closed_selected_addresses_keep_native_merlin_and_target_state(tmp_path, width):
    from test_mixed_effects import expected, qualify

    from t2m.bundle import _record
    from t2m.invocation import invoke_kernel, verify_invocation

    result = capture(tmp_path, width)
    reports = [
        r
        for r in result.trace["transformations"]
        if r["kind"] == "closed-pointer-integer-displacement"
    ]
    assert len(reports) == 2 and {r["element_bytes"] for r in reports} == {width // 8}
    assert {op["opcode"] for op in result.trace["source_operations"]} >= {
        "tt.ptr_to_int",
        "tt.int_to_ptr",
    }
    assert "prov.closed_pointer_addresses" in result.mlir_text
    from t2m.shared_tensor import opcode

    assert not any(
        opcode(op) in ("tt.ptr_to_int", "tt.int_to_ptr")
        for op in parse_shared(result.mlir_text).walk()
    )
    values, reference = arguments(width)
    policy = expected("", [{"kind": "words"}])
    bundle, compiled = qualify(result, tmp_path, values, policy, [reference])
    library = _record(compiled / "host/model.so")
    for index, options in enumerate(
        (
            {"n": 0, "origin": 8191, "capacity": 0},
            {"n": 1, "steps": 0, "origin": 8191, "capacity": 0},
            {"n": 19, "steps": 3, "origin": 64, "stride": -24},
            {"n": 31, "steps": 1, "origin": 40, "stride": 0},
        )
    ):
        values, reference = arguments(width, **options)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(values)}
        )
        np.savez(caller / "reference.npz", out0=reference)
        (caller / "reference.json").write_text(json.dumps(policy))
        invoke_kernel(
            compiled,
            caller / "inputs.npz",
            caller / "reference.json",
            tmp_path / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(tmp_path / f"replay-{index}")
        assert _record(compiled / "host/model.so") == library
    from test_spike_model import TOOLS

    if TOOLS:
        from scripts import qualify_spike_effect as target

        options = {
            "merlin_root": os.environ["T2M_MERLIN_ROOT"],
            "python": os.environ["T2M_MERLIN_PYTHON"],
        }
        receipt = target.qualify(bundle, tmp_path / "target", **options)
        assert target.verify(tmp_path / "target", **options) == receipt


@pytest.mark.skipif(
    not (OPT and NATIVE), reason="existing shared/native tools required"
)
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_scalar_subtraction_reuses_logical_root_in_every_memory_profile(
    tmp_path, profile
):
    from t2m.examples.pointer_address_kernels import scalar_displacement
    from t2m.frontend import capture_ttir
    from t2m.native import execute_native

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            scalar_displacement, {"x": "*i32", "out": "*i32", "offset": "i32"}, {}
        )
    )
    abi = TensorABI(
        "scalar_displacement",
        (
            TensorBuffer("x", (16,), dtype="i32"),
            TensorBuffer("out", (1,), "write", "i32"),
        ),
        runtime_scalars=(RuntimeScalar("offset", "i32", 2, bounds=(-15, 0)),),
    )
    result = convert_shared(
        source,
        abi,
        tmp_path / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    values = np.arange(16, dtype=np.int32) * 7 - 29
    for index, offset in enumerate((0, -5, -15)):
        actual, _ = execute_native(
            result.mlir_text,
            [values, np.asarray(offset, np.int32)],
            tmp_path / f"native-{index}",
            python=NATIVE,
            shape_abi=result.spec.runtime_shapes(),
        )
        np.testing.assert_array_equal(actual[0], values[-offset : -offset + 1])


@pytest.mark.skipif(not REQUIRED, reason="existing native/Merlin tools required")
@pytest.mark.parametrize("profile", ["tensor", "buffer", "guarded-buffer"])
def test_masked_scalar_gather_and_scatter_preserve_inactive_state(tmp_path, profile):
    from test_mixed_effects import expected, qualify

    from t2m.bundle import _record
    from t2m.examples.pointer_address_kernels import masked_scalar_memory
    from t2m.frontend import capture_ttir
    from t2m.invocation import invoke_kernel, verify_invocation

    source = tmp_path / "source.ttir"
    source.write_text(
        capture_ttir(
            masked_scalar_memory,
            {
                "x": "*i32",
                "out": "*i32",
                "offset": "i32",
                "destination": "i32",
                "active": "i1",
            },
            {},
        )
    )
    abi = TensorABI(
        "masked_scalar_memory",
        (
            TensorBuffer("x", (16,), dtype="i32"),
            TensorBuffer("out", (16,), "readwrite", "i32"),
        ),
        runtime_scalars=(
            RuntimeScalar("offset", "i32", 2, bounds=(-8192, 8192)),
            RuntimeScalar("destination", "i32", 3, bounds=(0, 15)),
            RuntimeScalar("active", "i1", 4),
        ),
    )
    result = convert_shared(
        source,
        abi,
        tmp_path / "translation",
        optimizer=OPT,
        memory_profile=profile,
        address_policy="source-words" if profile == "guarded-buffer" else "structured",
    )
    x = np.arange(16, dtype=np.int32) * 7 - 29
    initial = np.arange(16, dtype=np.int32) * 13 - 77

    def values(offset, destination, active):
        reference = initial.copy()
        if active:
            reference[destination] = x[offset]
        return [
            x,
            initial,
            np.asarray(offset, np.int32),
            np.asarray(destination, np.int32),
            np.asarray(active),
        ], reference

    arrays, reference = values(5, 3, True)
    policy = expected("", [{"kind": "words"}])
    _, compiled = qualify(result, tmp_path, arrays, policy, [reference])
    library = _record(compiled / "host/model.so")
    for index, options in enumerate(
        ((-8191, 0, False), (8191, 15, False), (15, 0, True))
    ):
        arrays, reference = values(*options)
        caller = tmp_path / f"caller-{index}"
        caller.mkdir()
        np.savez(
            caller / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)}
        )
        np.savez(caller / "reference.npz", out0=reference)
        (caller / "reference.json").write_text(json.dumps(policy))
        invoke_kernel(
            compiled,
            caller / "inputs.npz",
            caller / "reference.json",
            tmp_path / f"replay-{index}",
            merlin_root=os.environ["T2M_MERLIN_ROOT"],
            python=os.environ["T2M_MERLIN_PYTHON"],
        )
        verify_invocation(tmp_path / f"replay-{index}")
        assert _record(compiled / "host/model.so") == library


def _module(body):
    return parse_shared(
        """builtin.module { "tt.func"() <{sym_name="address", function_type=(!tt.ptr<i32>,!tt.ptr<i64>,i64) -> ()}> ({
      ^bb0(%p:!tt.ptr<i32>,%out:!tt.ptr<i64>,%offset:i64):
      %bits="tt.ptr_to_int"(%p):(!tt.ptr<i32>)->i64 loc("ttir.op.base")
      """
        + body
        + """
      "tt.return"():()->()
    }):()->() }"""
    )


@pytest.mark.parametrize(
    "body,message",
    [
        (
            '"tt.store"(%out,%bits):(!tt.ptr<i64>,i64)->() loc("ttir.op.escape")',
            "observe or escape",
        ),
        (
            '%sum=arith.addi %bits,%offset:i64 loc("ttir.op.add")\n%ptr="tt.int_to_ptr"(%sum):(i64)->!tt.ptr<i32> loc("ttir.op.cast")',
            "divisible",
        ),
        ('%neg=arith.subi %offset,%bits:i64 loc("ttir.op.neg")', "one base"),
        ('%twice=arith.addi %bits,%bits:i64 loc("ttir.op.double")', "one base"),
        (
            '%ptr="tt.int_to_ptr"(%offset):(i64)->!tt.ptr<i32> loc("ttir.op.inbound")',
            "original pointer base",
        ),
        (
            '%short=arith.trunci %bits:i64 to i32 loc("ttir.op.trunc")',
            "observe or escape",
        ),
        (
            '%sum="arith.addi"(%bits,%offset)<{overflowFlags=#arith.overflow<nsw>}>:(i64,i64)->i64 loc("ttir.op.poison")',
            "non-poisoning",
        ),
    ],
)
def test_address_observations_and_unproved_offsets_refuse_before_mutation(
    body, message
):
    from t2m.ir import text

    module = _module(body)
    before = text(module)
    values = [
        value
        for op in module.walk()
        for value in (
            *op.results,
            *(
                a
                for region in op.regions
                for block in region.blocks
                for a in block.args
            ),
        )
    ]
    uses = {v: set(v.uses) for v in values}
    with pytest.raises(ConversionError, match=message) as error:
        normalize_pointer_addresses(module, _source_ids)
    assert error.value.source_id and error.value.opcode
    assert text(module) == before and {v: set(v.uses) for v in values} == uses


def test_modular_byte_displacement_has_no_base_address_assumption():
    # Independent bit-vector identity over extreme source addresses/deltas.
    # No particular runtime root address or absence of unsigned wrap is assumed.
    mask = (1 << 64) - 1
    for width in (1, 2, 4, 8):
        for base in (0, 1, (1 << 63) - 1, 1 << 63, mask - 3, mask):
            for words in (
                -(1 << 63),
                -(1 << 62),
                -129,
                -1,
                0,
                1,
                129,
                (1 << 62) - 1,
                (1 << 63) - 1,
            ):
                displacement = (words * width) & mask
                signed = (
                    displacement - (1 << 64) if displacement >> 63 else displacement
                )
                logical = signed // width
                assert (base + displacement) & mask == (base + logical * width) & mask


def test_divisibility_proof_has_an_explicit_source_graph_budget():
    parts = ["%four=arith.constant 4:i64", "%v0=arith.muli %offset,%four:i64"]
    parts += [f"%v{i + 1}=arith.addi %v{i},%four:i64" for i in range(130)]
    parts += [
        "%sum=arith.addi %bits,%v130:i64",
        '%ptr="tt.int_to_ptr"(%sum):(i64)->!tt.ptr<i32> loc("ttir.op.budget")',
    ]
    module = _module("\n".join(parts))
    with pytest.raises(ConversionError, match="bounded SSA budget") as error:
        normalize_pointer_addresses(module, _source_ids)
    assert error.value.source_id == "ttir.op.budget"


def test_reorderable_address_shapes_need_a_paired_lane_contract():
    module = _module("""
      %tile="tt.splat"(%bits):(i64)->tensor<1xi64>
      %reshape="tt.reshape"(%tile)<{allow_reorder=unit}>:(tensor<1xi64>)->tensor<1xi64> loc("ttir.op.reshape")
      %ptr="tt.int_to_ptr"(%reshape):(tensor<1xi64>)->tensor<1x!tt.ptr<i32>>
    """)
    with pytest.raises(ConversionError, match="paired lane ordering") as error:
        normalize_pointer_addresses(module, _source_ids)
    assert error.value.source_id == "ttir.op.reshape"
