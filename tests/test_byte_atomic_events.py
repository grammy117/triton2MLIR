"""Actual upstream word templates retain a source byte-address boundary."""

import os
from copy import deepcopy

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.ir import Block, Region

from t2m.byte_atomic_events import check_prefix, prefix
from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.shared_tensor import _source_ids, opcode, parse_shared
from t2m.upstream import lower_with_shared
from t2m.word_events import convert_events, print_ir, substitute, templates

OPT = os.environ.get("TRITON_SHARED_OPT")
NATIVE = os.environ.get("T2M_NATIVE_PYTHON")


def source(width=32, cas=False, shape=(8,), masked=True):
    def tile(element):
        return f"tensor<{'x'.join(map(str, shape))}x{element}>" if shape else element

    pointer, byte = tile(f"!tt.ptr<i{width}>"), tile("!tt.ptr<i8>")
    offset, word, predicate = tile("i64"), tile(f"i{width}"), tile("i1")
    args = f"%base: !tt.ptr<i8>, %offset: {offset}, %value: {word}"
    if cas:
        args += f", %desired: {word}"
    elif masked:
        args += f", %mask: {predicate}"
    splat = f"%splat = tt.splat %base : !tt.ptr<i8> -> {byte}" if shape else ""
    if cas:
        event = f"%old = tt.atomic_cas acq_rel, gpu, %word, %value, %desired : ({pointer}, {word}, {word}) -> {word}"
    else:
        operands = f"%word, %value{', %mask' if masked else ''}"
        types = f"{pointer}, {word}{', ' + predicate if masked else ''}"
        event = f"%old = tt.atomic_rmw add, relaxed, gpu, {operands} : ({types}) -> {word}"
    return f"""module {{
      tt.func public @byte_atomic({args}) -> {word} {{
        {splat}
        %bytes = tt.addptr {'%splat' if shape else '%base'}, %offset : {byte}, {offset}
        %word = tt.bitcast %bytes : {byte} -> {pointer}
        {event}
        tt.return %old : {word}
      }}
    }}"""


def native_source(tmp_path, **kwargs):
    path = tmp_path / "source.ttir"
    path.write_text(source(**kwargs))
    lower_with_shared(path, tmp_path / "verified-source", optimizer=OPT, stage="source")
    return parse_shared((tmp_path / "verified-source/lowered.mlir").read_text())


@pytest.mark.skipif(not OPT, reason="existing pinned shared optimizer required")
@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize("shape", ((), (8,), (2, 4)))
@pytest.mark.parametrize("width,cas", ((32, False), (64, False), (16, True), (32, True), (64, True)))
def test_actual_native_templates_keep_one_word_event_and_byte_address_units(
    tmp_path, width, cas, shape, order
):
    module = native_source(tmp_path, width=width, cas=cas, shape=shape)
    before = print_ir(module)
    isolated, bindings, calls = templates(module, _source_ids, byte_order=order)
    assert print_ir(module) == before
    record, = bindings
    assert record["byte_view"]["address_units"] == "signed-i64-bytes"
    assert record["input_types"][0] == "!tt.ptr<i8>"
    assert record["template_input_types"][0] == f"!tt.ptr<i{width}>"
    path = tmp_path / "events.mlir"
    path.write_text(print_ir(isolated))
    report = lower_with_shared(path, tmp_path / "native", optimizer=OPT, stage="source-word-events")
    assert report["stage_completed"]
    converted = parse_shared((tmp_path / "native/lowered.mlir").read_text())
    body = next(iter(converted.ops)).regions[0].block
    check_prefix(body, record["byte_view"], not cas)
    memories = [op for op in body.walk() if opcode(op) in ("tt.atomic_cas", "tts.unstructured_atomic_rmw")]
    assert len(memories) == 1
    substitute(module, converted, bindings, calls)
    assert sum(opcode(op) in ("tt.atomic_cas", "tts.unstructured_atomic_rmw") for op in module.walk()) == 1
    assert not any(opcode(op) == "tt.atomic_rmw" for op in module.walk())


@pytest.mark.skipif(not NATIVE, reason="existing native MLIR interpreter required")
@pytest.mark.parametrize("shape", ((), (4,), (2, 2)))
@pytest.mark.parametrize("masked", (False, True))
def test_active_alignment_executes_before_division_and_ignores_inactive_extremes(
    tmp_path, shape, masked
):
    from t2m.native import execute_native, observe_native

    offset_type = builtin.TensorType(builtin.i64, shape) if shape else builtin.i64
    mask_type = builtin.TensorType(builtin.i1, shape) if shape else builtin.i1
    # The private pointer formal is unused by this pure alignment/index test.
    body = Block(arg_types=[builtin.i64, offset_type, *([mask_type] if masked else [])])
    indices = prefix(body, 4, body.args[-1] if masked else None, builtin.UnknownLoc())
    result_type = offset_type
    if not shape:
        wrapped = tensor.FromElementsOp(indices)
        body.add_op(wrapped)
        indices, result_type = wrapped.result, wrapped.result.type
    body.add_op(func.ReturnOp(indices))
    module = builtin.ModuleOp([func.FuncOp("alignment", ([arg.type for arg in body.args], [result_type]), Region(body))])
    values = np.asarray([-(2**63), -8, 0, 12], np.int64).reshape(shape) if shape else np.asarray(-8, np.int64)
    flags = np.ones(shape, np.bool_)
    if masked and shape:
        values.flat[0] = -(2**63) + 1
        flags.flat[0] = False
    arrays = [np.asarray(0, np.int64), values, *([flags] if masked else [])]
    (actual,), _ = execute_native(text(module), arrays, tmp_path / "valid", python=NATIVE)
    # Native integer division truncates toward zero, including unused lanes.
    expected = np.asarray([int(x) // 4 if int(x) >= 0 else -(-int(x) // 4) for x in values.flat], np.int64).reshape(shape)
    np.testing.assert_array_equal(actual, expected if shape else expected.reshape(1))
    values.flat[-1] = 3
    outcome = observe_native(text(module), arrays, tmp_path / "invalid", python=NATIVE)
    assert outcome["returncode"] == -6
    assert (tmp_path / "invalid/stdout.bin").read_bytes() == b"Triton byte atomic address is not word aligned\n"


@pytest.mark.skipif(not OPT, reason="existing pinned shared optimizer required")
def test_changed_alignment_or_divisor_does_not_pass_current_body_reconstruction(tmp_path):
    module = native_source(tmp_path)
    isolated, bindings, _ = templates(module, _source_ids, byte_order="little")
    for kind in ("divisor", "assertion", "mask"):
        changed = isolated.clone()
        body = next(iter(changed.ops)).regions[0].block
        if kind == "divisor":
            divide = next(op for op in body.walk() if isinstance(op, arith.DivSIOp))
            divide.operands = [divide.operands[0], divide.operands[0]]
        elif kind == "assertion":
            assertion = next(op for op in body.walk() if op.name == "cf.assert")
            assertion.parent.erase_op(assertion)
        else:
            branch = next(op for op in body.walk() if op.name == "scf.if")
            compare = arith.XOrIOp(branch.cond, branch.cond)
            branch.parent.insert_op_before(compare, branch)
            branch.operands = [compare.result]
        with pytest.raises(ConversionError, match="active alignment|byte-to-word"):
            check_prefix(body, bindings[0]["byte_view"], True)


@pytest.mark.skipif(not OPT, reason="existing pinned shared optimizer required")
@pytest.mark.parametrize("cas", (False, True))
@pytest.mark.parametrize("change", ("offset", "value", "old", "bytes", "mask-or-comparison"))
def test_substitution_refuses_changed_native_boundary_before_parent_edits(tmp_path, cas, change):
    module = native_source(tmp_path, cas=cas)
    isolated, bindings, calls = templates(module, _source_ids, byte_order="little")
    path = tmp_path / "events.mlir"
    path.write_text(print_ir(isolated))
    lower_with_shared(path, tmp_path / "native", optimizer=OPT, stage="source-word-events")
    converted = parse_shared((tmp_path / "native/lowered.mlir").read_text())
    body = next(iter(converted.ops)).regions[0].block
    event = next(op for op in body.walk() if opcode(op) == ("tt.atomic_cas" if cas else "tts.unstructured_atomic_rmw"))
    if change == "offset":
        if cas:
            descriptor = event.operands[0].owner
            descriptor.operands = [descriptor.operands[0], body.args[1]]
        else:
            event.operands = [event.operands[0], body.args[1], *event.operands[2:]]
    elif change == "value":
        operands = list(event.operands)
        operands[2] = operands[1] if cas else body.args[1]
        event.operands = operands
    elif change == "old":
        body.last_op.operands = [body.args[2]]
    elif change == "bytes":
        bindings[0]["byte_view"]["alignment_bytes"] *= 2
    elif cas:
        event.operands = [event.operands[0], event.operands[2], event.operands[1]]
    else:
        event.operands = list(event.operands[:-1])
    before = print_ir(module)
    with pytest.raises(ConversionError, match="byte atomic|byte storage"):
        substitute(module, converted, bindings, calls)
    assert print_ir(module) == before


@pytest.mark.skipif(not OPT, reason="existing pinned shared optimizer required")
@pytest.mark.parametrize("width,cas,scalar", ((32, False, False), (64, False, True), (16, True, False), (64, True, True)))
@pytest.mark.parametrize("order", ("little", "big"))
def test_real_triton_capture_replays_both_native_stages_with_byte_boundary(tmp_path, width, cas, scalar, order):
    from t2m.examples.byte_atomic_kernels import byte_atomic_addresses
    from t2m.frontend import capture_ttir
    from t2m.word_evidence import check_word_stage, record, replay_events

    original = tmp_path / "source.ttir"
    original.write_text(capture_ttir(byte_atomic_addresses, {
        "raw": "*i8", "offsets": "*i64", "values": f"*i{width}", "old": f"*i{width}", "n": "i64",
        "B": "constexpr", "WIDTH": "constexpr", "CAS": "constexpr", "SCALAR": "constexpr",
    }, {"B": 8, "WIDTH": width, "CAS": cas, "SCALAR": scalar}))
    lower_with_shared(original, tmp_path / "source", optimizer=OPT, stage="source")
    source_bytes = (tmp_path / "source/lowered.mlir").read_text()
    module = parse_shared(source_bytes)
    events = convert_events(module, tmp_path / "events", OPT, _source_ids, byte_order=order)
    selected = [binding for binding in events["bindings"] if "byte_view" in binding]
    assert len(selected) == 1 and events["byte_order"] == order
    compute = tmp_path / "compute.mlir"
    compute.write_text(print_ir(module))
    native = lower_with_shared(compute, tmp_path / "compute", optimizer=OPT, stage="source-word-compute")
    output = (tmp_path / "compute/lowered.mlir").read_text()
    stage = {"schema": "t2m.source-word-stages.v2", "stage_completed": True,
             "reconstruction": "same-region-same-event-order", "input": record(source_bytes), "output": record(output),
             "events": events, "compute": {"input": compute.read_text(), "native_stage": native}}
    check_word_stage(stage, source_bytes, output)
    replay_events(stage, source_bytes)
    for mutation in ("missing-order", "order-disagrees", "binding", "division"):
        changed = deepcopy(stage)
        if mutation == "missing-order":
            del changed["events"]["byte_order"]
        elif mutation == "order-disagrees":
            changed["events"]["byte_order"] = "big" if order == "little" else "little"
        elif mutation == "binding":
            chosen = next(binding for binding in changed["events"]["bindings"] if "byte_view" in binding)
            chosen["input_types"][0] = f"!tt.ptr<i{width}>"
        else:
            converted = parse_shared(changed["events"]["output"])
            divide = next(op for op in converted.walk() if isinstance(op, arith.DivSIOp))
            divide.operands = [divide.operands[0], divide.operands[0]]
            changed["events"]["output"] = print_ir(converted)
            changed["events"]["native_stage"]["artifacts"]["lowered.mlir"] = record(changed["events"]["output"])
        with pytest.raises((ValueError, ConversionError)):
            check_word_stage(changed, source_bytes, output)
            replay_events(changed, source_bytes)


@pytest.mark.skipif(not OPT, reason="existing pinned shared optimizer required")
def test_byte_templates_require_explicit_order_without_parent_edits(tmp_path):
    module = native_source(tmp_path)
    before = print_ir(module)
    for order in (None, "native", False):
        with pytest.raises(ConversionError, match="explicit little or big"):
            templates(module, _source_ids, byte_order=order)
        assert print_ir(module) == before


@pytest.mark.skipif(not OPT, reason="existing pinned shared optimizer required")
def test_word_addptr_after_cast_cannot_be_reinterpreted_as_byte_units(tmp_path):
    original = source(shape=())
    original = original.replace(
        "%bytes = tt.addptr %base, %offset : !tt.ptr<i8>, i64\n        %word = tt.bitcast %bytes : !tt.ptr<i8> -> !tt.ptr<i32>",
        "%cast = tt.bitcast %base : !tt.ptr<i8> -> !tt.ptr<i32>\n        %word = tt.addptr %cast, %offset : !tt.ptr<i32>, i64",
    )
    path = tmp_path / "source.ttir"
    path.write_text(original)
    lower_with_shared(path, tmp_path / "native", optimizer=OPT, stage="source")
    module = parse_shared((tmp_path / "native/lowered.mlir").read_text())
    before = print_ir(module)
    with pytest.raises(ConversionError, match="equal-width logical storage"):
        templates(module, _source_ids, byte_order="little")
    assert print_ir(module) == before
