"""Typed entry ports require complete source bindings and checked storage."""

from copy import deepcopy
from dataclasses import replace

import pytest
from xdsl.dialects import builtin
from xdsl.ir import Block, Region

from t2m.allocation_bindings import (
    PointerArgument,
    ResultBinding,
    ValueArgument,
    arguments_as_dict,
    arguments_from_dict,
    bind_module,
    entry_arguments_from_dict,
)
from t2m.diagnostics import ConversionError
from t2m.ir import text
from t2m.shared_tensor import TensorABI, TensorBuffer


def case():
    body = Block(arg_types=[builtin.bf16])
    body.add_op(builtin.UnregisteredOp.with_name("tt.return").create(operands=[body.args[0], body.args[0]]))
    function = builtin.UnregisteredOp.with_name("tt.func").create(
        properties={"sym_name": builtin.StringAttr("values"), "sym_visibility": builtin.StringAttr("public"),
            "function_type": builtin.FunctionType.from_lists([builtin.bf16], [builtin.bf16, builtin.bf16])},
        regions=[Region(body)])
    module = builtin.ModuleOp([function])
    abi = TensorABI("bound", (TensorBuffer("x", (2,), dtype="bf16"),
        TensorBuffer("first", (2,), "write", "bf16"), TensorBuffer("second", (2,), "write", "bf16")), grid=(2,))
    return module, abi, (ValueArgument("x", per_program=True),), (ResultBinding("first"), ResultBinding("second"))


def test_value_ports_roundtrip_without_discarding_results_or_mutating_source():
    module, abi, inputs, results = case()
    configuration = arguments_as_dict("values", inputs, result_bindings=results)
    entry, configured, returned = entry_arguments_from_dict(configuration)
    assert (entry, configured, returned) == ("values", inputs, results)
    with pytest.raises(ValueError, match="never|result ports|result_bindings"):
        arguments_from_dict(configuration)
    before = text(module)
    wrapped, report = bind_module(module, entry, abi, configured, result_bindings=returned)
    assert text(module) == before and wrapped is not module
    assert report["grants_semantics"] is False
    assert len(report["result_bindings"]) == 2


@pytest.mark.parametrize("change", ("missing-input", "wrong-input-type", "write-input", "unknown-origin",
    "missing-result", "extra-result", "reused-result", "readonly-result", "wrong-result-type", "unused-root"))
def test_bad_value_storage_and_incomplete_rosters_refuse(change):
    module, abi, inputs, results = case()
    if change == "missing-input":
        inputs = (ValueArgument("absent"),)
    elif change == "wrong-input-type":
        abi = replace(abi, buffers=(replace(abi.buffers[0], dtype="i16"), *abi.buffers[1:]))
    elif change == "write-input":
        inputs = (ValueArgument("first"),)
    elif change == "unknown-origin":
        inputs = (ValueArgument("x", "absent"),)
    elif change == "missing-result":
        results = results[:1]
    elif change == "extra-result":
        results = (*results, results[0])
    elif change == "reused-result":
        results = (results[0], results[0])
    elif change == "readonly-result":
        results = (ResultBinding("x"), results[1])
    elif change == "wrong-result-type":
        abi = replace(abi, buffers=(abi.buffers[0], replace(abi.buffers[1], dtype="i16"), abi.buffers[2]))
    else:
        abi = replace(abi, buffers=(*abi.buffers, TensorBuffer("unused", (2,), dtype="bf16")))
    with pytest.raises(ConversionError):
        bind_module(module, "values", abi, inputs, result_bindings=results)


@pytest.mark.parametrize("port", (ValueArgument, ResultBinding))
@pytest.mark.parametrize("fields", ({"offset": True}, {"offset": 2**63}, {"offset": 0.5},
    {"per_program": 1}, {"allocation": "bad root"}))
def test_value_port_fields_are_exact_types(port, fields):
    with pytest.raises(ValueError):
        port(**{"allocation": "x", **fields})


@pytest.mark.parametrize("change", ("source-type", "source-root-type", "result-type", "result-root-type",
    "missing-result", "missing-result-report", "origin", "Boolean-origin", "program-mode", "schema", "unknown-field", "value-result-alias"))
def test_consumer_checks_source_types_and_rehashed_port_rosters_independently(change):
    from merlin.compiler.frontend.intake.kernel_entry_values import checked_entry_values

    module, abi, inputs, results = case()
    _, binding = bind_module(module, "values", abi, inputs, result_bindings=results)
    configuration = arguments_as_dict("values", inputs, result_bindings=results)
    spec = deepcopy(abi.as_dict())
    checked_entry_values(configuration, binding, spec, text(module).encode())
    if change == "source-type":
        binding["bindings"][0]["source_type"] = "i16"
    elif change == "source-root-type":
        spec["buffers"][0]["dtype"] = "i16"
        binding["bindings"][0]["source_type"] = "i16"
    elif change == "result-type":
        binding["result_bindings"][0]["source_type"] = "i16"
    elif change == "result-root-type":
        spec["buffers"][1]["dtype"] = "i16"
        binding["result_bindings"][0]["source_type"] = "i16"
    elif change == "missing-result":
        configuration["results"].pop()
    elif change == "missing-result-report":
        binding.pop("result_bindings")
    elif change == "origin":
        configuration["arguments"][0]["offset"] = 1
    elif change == "Boolean-origin":
        configuration["arguments"][0]["offset"] = True
    elif change == "program-mode":
        configuration["results"][0]["per_program"] = "yes"
    elif change == "schema":
        configuration["schema"] = "t2m.source-allocation-arguments.v1"
    elif change == "value-result-alias":
        spec["buffers"][0]["effect"] = "readwrite"
        configuration["results"][0]["allocation"] = "x"
        binding["result_bindings"][0]["allocation"] = "x"
    else:
        configuration["grants_semantics"] = True
    with pytest.raises(ValueError):
        checked_entry_values(configuration, binding, spec, text(module).encode())


def test_result_port_cannot_overwrite_original_pointer_state():
    from t2m.ir import context
    from t2m.shared_tensor import SharedParser

    ctx = context()
    ctx.allow_unregistered = True
    pointer = SharedParser(ctx, "!tt.ptr<i32>").parse_type()
    body = Block(arg_types=[pointer])
    zero = builtin.UnregisteredOp.with_name("arith.constant").create(result_types=[builtin.i32],
        properties={"value": builtin.IntegerAttr(0, builtin.i32)})
    body.add_ops([zero, builtin.UnregisteredOp.with_name("tt.return").create(operands=[zero.results[0]])])
    function = builtin.UnregisteredOp.with_name("tt.func").create(properties={"sym_name": builtin.StringAttr("pointer"),
        "function_type": builtin.FunctionType.from_lists([pointer], [builtin.i32])}, regions=[Region(body)])
    abi = TensorABI("bound_pointer", (TensorBuffer("state", (1,), "readwrite", "i32"),))
    with pytest.raises(ConversionError, match="separate storage"):
        bind_module(builtin.ModuleOp([function]), "pointer", abi, (PointerArgument("state"),),
            result_bindings=(ResultBinding("state"),))


def test_value_input_cannot_reload_a_later_programs_result():
    module, abi, inputs, _ = case()
    abi = replace(abi, buffers=(replace(abi.buffers[0], effect="readwrite"), abi.buffers[2]))
    with pytest.raises(ConversionError, match="value inputs need storage separate"):
        bind_module(module, "values", abi, inputs,
            result_bindings=(ResultBinding("x"), ResultBinding("second")))
